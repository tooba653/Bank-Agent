import os
import asyncio
from typing import Any
from pydantic import BaseModel
from dotenv import load_dotenv

from agents import (
    Agent,
    RunConfig,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    Runner,
    function_tool,
    RunContextWrapper,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    handoffs,
    input_guardrail,
    output_guardrail,
)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in .env file.")

provider = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider,
)

run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=False,
)

class Account(BaseModel):
    name: str
    pin: int

class MessageOutput(BaseModel):
    response: str

class GuardrailAgentOutput(BaseModel):
    is_bank_related: bool
    reasoning: str | None = None

class OutputCheck(BaseModel):
    is_bank_queries: bool
    reasoning: str | None = None

loan_agent = Agent(
    name="Loan Agent",
    instructions="You specialize in loan-related questions. Share details about loan types, requirements, interest, repayment terms, and general application steps.",
)

customer_agent = Agent(
    name="Customer Agent",
    instructions="You handle customer service queries such as deposits, withdrawals, refunds, and authentication issues. Be polite and ask for any missing details when necessary.",
)

guardrail_agent = Agent(
    name="Input Guardrail: Banking Relevance Check",
    instructions="Decide if the user’s input is about banking topics (e.g., balance, transfer, withdrawal, deposit, loans, refunds, authentication). Output a boolean 'is_bank_related' and a short explanation.",
    output_type=GuardrailAgentOutput,
)

@input_guardrail
async def check_bank_related(ctx: RunContextWrapper[None], agent: Agent, input: str) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not bool(result.final_output.is_bank_related),
    )


def is_authenticated(ctx: RunContextWrapper[Account], agent: Agent) -> bool:
    try:
        return ctx.context.name == "Tooba" and ctx.context.pin == 678
    except Exception:
        return False

@function_tool(is_enabled=is_authenticated)
def check_balance(account_number: str) -> str:
    return f"The balance for account {account_number} is $100,000"

control_guardrail_agent = Agent(
    name="Output Guardrail: Banking Safety Filter",
    instructions="Review the agent’s response. Confirm it is only about valid banking topics and contains no unsafe or irrelevant content. Return a boolean 'is_bank_queries' with reasoning.",
    output_type=OutputCheck,
)

@output_guardrail
async def control_response(ctx: RunContextWrapper[None], agent: Agent, output: MessageOutput) -> GuardrailFunctionOutput:
    result = await Runner.run(control_guardrail_agent, output.response, context=ctx.context)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not bool(result.final_output.is_bank_queries),
    )

bank_agent = Agent(
    name="Bank Agent",
    instructions="You are a digital bank assistant. Manage balance checks, deposits, withdrawals, transfers, and refunds. Always verify user identity before sharing sensitive details. If the request is more specific, pass it to customer_agent or loan_agent.",
    handoffs=[customer_agent, loan_agent],
    input_guardrails=[check_bank_related],
    output_guardrails=[control_response],
    output_type=MessageOutput,
)

user_context = Account(name="Tooba", pin=678)


async def main():
    try:
        result = await Runner.run(bank_agent, "I want to check my balance.", context=user_context, run_config=run_config)
        print("Agent result:", result.final_output.response)
    except InputGuardrailTripwireTriggered:
        print("Input guardrail tripped: user input deemed not bank-related.")
    except OutputGuardrailTripwireTriggered:
        print("Output guardrail tripped: agent produced non-bank or unsafe output.")

    try:
        await Runner.run(bank_agent, "Help me solve my math homework: 6x + 9 = 22", context=user_context, run_config=run_config)
        print("Unexpected: guardrail didn't trip for non-bank input.")
    except InputGuardrailTripwireTriggered:
        print("Correctly trapped: non-bank input detected by input guardrail.")
    except OutputGuardrailTripwireTriggered:
        print("Output guardrail tripped (unexpected for this test).")

if __name__ == "__main__":
    asyncio.run(main())

