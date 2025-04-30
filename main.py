import asyncio
import io
import re
from contextlib import redirect_stdout
from datetime import datetime, timedelta

import requests
from ollama import AsyncClient

from config import get_settings

settings = get_settings()

MODEL = "gemma3:4b"


# extract the tool call from the response
def extract_tool_call(text):
    pattern = r"```tool_code\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        # Capture stdout in a string buffer
        f = io.StringIO()
        with redirect_stdout(f):
            result = eval(code)
        output = f.getvalue()
        r = result if output == "" else output
        return f"```tool_output\n{str(r).strip()}\n```"
    return None


def get_current_date_time() -> str:
    """
    Gets the current system time and formats it as a string.

    Returns:
        str: The current system time formatted as Weekday, Month Day, YYYY HH:MM:SS.
    """
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y %H:%M:%S")


def convert(amount: float, currency: str, new_currency: str) -> None | float:
    # default ask:
    ask: float = 1.0

    # date today:
    date_today = datetime.now().strftime("%Y-%m-%d")

    # date yesterday:
    date_yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # generate the url:
    url = f"https://{settings.EXCHANGE_RATE_SITE}/cc-api/currencies?base={currency}&quote={new_currency}&data_type=general_currency_pair&start_date={date_yesterday}&end_date={date_today}"

    response = requests.request("GET", url)

    # convert to json:
    rates = response.json()

    if not rates:
        return None

    if rates:
        if "error" in rates:
            return None

    for rate in rates["response"]:
        ask = rate["average_ask"]
        break

    return float(ask) * amount


def get_current_exchange_rate(currency: str, new_currency: str) -> None | float:
    # default ask:
    ask: float = 1.0

    # date today:
    date_today = datetime.now().strftime("%Y-%m-%d")

    # date yesterday:
    date_yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # generate the url:
    url = f"https://{settings.EXCHANGE_RATE_SITE}/cc-api/currencies?base={currency}&quote={new_currency}&data_type=general_currency_pair&start_date={date_yesterday}&end_date={date_today}"

    response = requests.request("GET", url)

    # convert to json:
    rates = response.json()

    if not rates:
        return None

    if rates:
        if "error" in rates:
            return None

    for rate in rates["response"]:
        ask = rate["average_ask"]
        break

    return float(ask)


def get_historical_exchange_rate(
    currency: str, new_currency: str, date: str
) -> None | float:
    # default ask:
    ask: float = 1.0

    # conversion date:
    d = datetime.strptime(date, "%Y-%m-%d")

    # date yesterday:
    previous_day = (d - timedelta(days=1)).strftime("%Y-%m-%d")

    # generate the url:
    url = f"https://{settings.EXCHANGE_RATE_SITE}/cc-api/currencies?base={currency}&quote={new_currency}&data_type=general_currency_pair&start_date={previous_day}&end_date={date}"

    response = requests.request("GET", url)

    # convert to json:
    rates = response.json()

    if not rates:
        return None

    if rates:
        if "error" in rates:
            return None

    for rate in rates["response"]:
        ask = rate["average_ask"]
        break

    return float(ask)


# Ensure you do not include any "." in the prompt - you will get errors during the function call!S
instruction_prompt = '''You are a helpful conversational AI assistant.
At each turn, if you decide to invoke any of the function(s), it should be wrapped with ```tool_code```.
The python methods described below are imported and available, you can only use defined methods.
ONLY use the ```tool_code``` format when absolutely necessary to answer the user's question.
The generated code should be readable and efficient.

For questions that don't require any specific tools, just respond normally without tool calls.

The response to a method will be wrapped in ```tool_output``` use it to call more tools or generate a helpful, friendly response.
When using a ```tool_call``` think step by step why and how it should be used.

The following Python methods are available:

```python
def get_current_date_time() -> str:
    """Gets the current system time and formats it as a string

    Args:
        None
    """

def convert(amount: float, currency: str, new_currency: str) -> float:
    """Convert the currency with the latest exchange rate

    Args:
      amount: The amount of currency to convert
      currency: The currency to convert from
      new_currency: The currency to convert to
    """

def get_current_exchange_rate(currency: str, new_currency: str) -> float:
    """Get the latest exchange rate for the currency pair

    Args:
      currency: The currency to convert from
      new_currency: The currency to convert to
    """

def get_historical_exchange_rate(currency: str, new_currency: str, date: str) -> float:
    """Get the historical exchange rate for the currency pair on a specific date

    Args:
      currency: The currency to convert from
      new_currency: The currency to convert to
      date: The target date (in 'YYYY-MM-DD' format) for which to fetch the rate
    """
```
'''


async def main():
    client = AsyncClient()
    messages = []  # Stores the conversation history

    print("Starting chat with local model. Type 'quit' to exit.")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Exiting chat..")
                break

            # Format the message: include instructions only on the first turn
            if not messages:
                messages.append({"role": "system", "content": instruction_prompt})

            messages.append({"role": "user", "content": user_input})

            print("\nAssistant: ", end="", flush=True)

            # --- First Call to LLM with streaming ---
            full_response = ""
            async for chunk in await client.chat(
                model=MODEL, messages=messages, stream=True
            ):
                if chunk.get("message", {}).get("content"):
                    content = chunk["message"]["content"]
                    print(content, end="", flush=True)
                    full_response += content

            # Store the complete response in messages
            assistant_message = full_response
            messages.append({"role": "assistant", "content": assistant_message})

            # Check if the response actually contains a tool call
            contains_tool_syntax = "```tool_code" in assistant_message

            if not contains_tool_syntax:
                # No tool call syntax, we've already printed the response
                # Add a newline
                print()
                continue

            # If we get here, there's a tool call syntax in the message
            print("\n(Executing tool...)")

            # --- Extract and Execute Tool Call ---
            tool_output = extract_tool_call(assistant_message)

            if tool_output:
                # If a tool was successfully called, send the output back to the model
                messages.append({"role": "user", "content": tool_output})

                print("\nAssistant: ", end="", flush=True)

                # --- Second Call to LLM with streaming (after tool execution) ---
                final_response = ""
                async for chunk in await client.chat(
                    model=MODEL, messages=messages, stream=True
                ):
                    if chunk.get("message", {}).get("content"):
                        content = chunk["message"]["content"]
                        print(content, end="", flush=True)
                        final_response += content

                # Store the complete final response
                messages.append({"role": "assistant", "content": final_response})
                print()  # Add a newline
            else:
                # The message had tool_code syntax but no valid tool was executed
                print("\n(No valid tool execution was needed)")

        except KeyboardInterrupt:
            print("\nExiting chat.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExecution interrupted. Goodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
