import asyncio
import io
import re
from contextlib import redirect_stdout
from datetime import datetime

from ollama import AsyncClient

MODEL = "gemma3:1b"


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
        "get_current_time": get_current_time,
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


def get_current_time() -> str:
    """
    Gets the current system time and formats it as a string including
    the weekday name and month name.

    Returns:
        str: The current system time formatted as
             Weekday, Month Day, YYYY HH:MM:SS.
    """
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y %H:%M:%S")


instruction_prompt = '''You are a helpful assistant that can respond to questions directly or use tools when needed.

ONLY use the ```tool_code``` format when absolutely necessary to answer the user's question.
For questions that don't require any specific tools, just respond normally without tool calls.

The following Python functions are available when needed:

```python
def get_current_time() -> str:
    """
    Gets the current system time and formats it as a string including
    the weekday name and month name.

    Returns:
        str: The current system time formatted as
             Weekday, Month Day, YYYY HH:MM:SS.
    """
```

When you need to use a tool, place the code in a ```tool_code``` block like this:
```tool_code
get_current_time()
```

The response will be returned in a ```tool_output``` block.
'''


async def main():
    client = AsyncClient()
    messages = []  # Stores the conversation history

    print("Starting chat with local model. Type 'quit' to exit.")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Exiting chat.")
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
