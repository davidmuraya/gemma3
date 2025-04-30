import asyncio
import io
import re
from contextlib import redirect_stdout
from datetime import datetime, timedelta

import pytz
from ollama import AsyncClient

MODEL = "gemma3:4b"


def extract_tool_call(text):
    """
    Parses the model's response to find a tool_code block,
    executes the code using eval() with a restricted namespace,
    and returns the output formatted as ```tool_output```.
    Returns None if no tool_code is found.
    """
    pattern = r"```tool_code\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)

    if not match:
        return None

    code = match.group(1).strip()

    # Only allow access to specific functions to improve safety
    safe_globals = {
        "get_current_date_time": get_current_date_time,
        # Add other safe functions here
    }

    # Check if the code is actually trying to call a function
    if not any(func_name in code for func_name in safe_globals):
        # Skip execution if no valid function call is detected
        return None

    try:
        # Capture stdout in a string buffer
        f = io.StringIO()
        with redirect_stdout(f):
            # Use a restricted namespace for evaluation
            result = eval(code, {"__builtins__": {}}, safe_globals)

        output = f.getvalue()
        r = result if output == "" else output
        return f"```tool_output\n{str(r).strip()}\n```"
    except Exception as e:
        return f"```tool_output\nError executing tool: {str(e)}\n```"


# Function to get the current system time
def get_current_date_time():
    """
    Get the current date and time in different time zones.

    Retrieves the current UTC time and computes the local time
    for each predefined time zone by applying fixed offsets. It then formats
    each local time as `YYYY-MM-DD HH:MM:SS` and compiles a line for each zone:
    `"The current date and time in <Location> (<UTC offset>, <TZ code>) is <timestamp>."`

    Returns:
        str: A single string containing one line per time zone, separated by
             newline characters. Each line reports the current local date and
             time for a specific region, including its UTC offset and
             time-zone abbreviation.
    """

    # Define time zones
    time_zones = [
        ("UTC +14", "LINT", "Kiritimati", timedelta(hours=14)),
        ("UTC +13:45", "CHADT", "Chatham Islands", timedelta(hours=13, minutes=45)),
        ("UTC +13", "NZDT", "Auckland", timedelta(hours=13)),
        ("UTC +12", "ANAT", "Anadyr", timedelta(hours=12)),
        ("UTC +11", "AEDT", "Melbourne", timedelta(hours=11)),
        ("UTC +10:30", "ACDT", "Adelaide", timedelta(hours=10, minutes=30)),
        ("UTC +10", "AEST", "Brisbane", timedelta(hours=10)),
        ("UTC +9:30", "ACST", "Darwin", timedelta(hours=9, minutes=30)),
        ("UTC +9", "JST", "Tokyo", timedelta(hours=9)),
        ("UTC +8:45", "ACWST", "Eucla", timedelta(hours=8, minutes=45)),
        ("UTC +8", "CST", "Beijing", timedelta(hours=8)),
        ("UTC +7", "WIB", "Jakarta", timedelta(hours=7)),
        ("UTC +6:30", "MMT", "Yangon", timedelta(hours=6, minutes=30)),
        ("UTC +6", "BST", "Dhaka", timedelta(hours=6)),
        ("UTC +5:45", "NPT", "Kathmandu", timedelta(hours=5, minutes=45)),
        ("UTC +5:30", "IST", "New Delhi", timedelta(hours=5, minutes=30)),
        ("UTC +5", "UZT", "Tashkent", timedelta(hours=5)),
        ("UTC +4:30", "AFT", "Kabul", timedelta(hours=4, minutes=30)),
        ("UTC +4", "GST", "Dubai", timedelta(hours=4)),
        ("UTC +3:30", "IRST", "Tehran", timedelta(hours=3, minutes=30)),
        ("UTC +3", "MSK", "Moscow", timedelta(hours=3)),
        ("UTC +2", "EET", "Cairo", timedelta(hours=2)),
        ("UTC +1", "CET", "Brussels", timedelta(hours=1)),
        ("UTC +0", "GMT", "London", timedelta(hours=0)),
        ("UTC -1", "CVT", "Praia", timedelta(hours=-1)),
        ("UTC -2", "WGT", "Nuuk", timedelta(hours=-2)),
        ("UTC -3", "ART", "Buenos Aires", timedelta(hours=-3)),
        ("UTC -3:30", "NST", "St. John's", timedelta(hours=-3, minutes=-30)),
        ("UTC -4", "VET", "Caracas", timedelta(hours=-4)),
        ("UTC -5", "EST", "New York", timedelta(hours=-5)),
        ("UTC -6", "CST", "Mexico City", timedelta(hours=-6)),
        ("UTC -7", "MST", "Calgary", timedelta(hours=-7)),
        ("UTC -8", "PST", "Los Angeles", timedelta(hours=-8)),
        ("UTC -9", "AKST", "Anchorage", timedelta(hours=-9)),
        ("UTC -9:30", "MART", "Taiohae", timedelta(hours=-9, minutes=-30)),
        ("UTC -10", "HST", "Honolulu", timedelta(hours=-10)),
        ("UTC -11", "NUT", "Alofi", timedelta(hours=-11)),
        ("UTC -12", "AoE", "Baker Island", timedelta(hours=-12)),
    ]

    now_utc = datetime.now(pytz.utc)
    time_strings = []

    for tz_name, code, location, offset in time_zones:
        local_time = now_utc + offset
        time_strings.append(
            f"The current date and time in {location} ({tz_name}, {code}) is {local_time.strftime('%Y-%m-%d %H:%M:%S')}."
        )

    return "\n".join(time_strings)


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
def get_current_date_time():
    """
    Get the current date and time in different time zones.

    Retrieves the current UTC time and computes the local time
    for each predefined time zone by applying fixed offsets. It then formats
    each local time as `YYYY-MM-DD HH:MM:SS` and compiles a line for each zone:
    `"The current date and time in <Location> (<UTC offset>, <TZ code>) is <timestamp>."`

    Args:
        None

    Returns:
        str: A single string containing one line per time zone, separated by
             newline characters. Each line reports the current local date and
             time for a specific region, including its UTC offset and
             time-zone abbreviation.
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
                print()  # Add a newline
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
