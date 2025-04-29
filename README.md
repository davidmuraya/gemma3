## Project Overview

This repository contains a single Python script, `main.py`, that provides a local conversational interface to an Ollama model (`gemma3:4b`) with built-in tool-calling capabilities. 

Gemma3 uses pythonic function calling. https://www.philschmid.de/gemma-function-calling

This is quite different approach from the OpenAI-style function calling with json schema.

You can read about pythonic function calling here: https://huggingface.co/blog/andthattoo/dria-agent-a

The script:

- Starts an interactive chat loop.
- Sends user prompts to the local Ollama API.
- Detects and executes embedded `tool_code` blocks in the model's responses.
- Returns tool execution output back into the conversation.

## File Structure

```
/  
├── main.py    # Chat client with tool-calling logic
└── README.md  # Project documentation
└── requirements.txt  # Python requirements
```

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/) running locally and listening on `http://127.0.0.1:11434`
- Gemma3 Model run `ollama run gemma3:4b` after installing ollama.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/davidmuraya/gemma3.git
   cd gemma3
   ```

2. (Optional) Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # on Windows use `venv\\Scripts\\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   > **Note:** Ensure `ollama` and `requests` are included in your `requirements.txt`.

## Usage

Run the chat client:

```bash
python main.py
```

You will see a prompt:

```
Starting chat with local model. Type 'quit' to exit.
You:
```

- Type your question and hit **Enter**.
- If the model responds with a `tool_code` block (e.g., to call `get_current_time()`), the script will execute the code and feed the output back to the model.
- Type `quit`, `exit`, or `q` to end the session.

## How It Works

1. **Conversation Loop**: The script collects messages in a `messages` list and streams responses from the Ollama API.
2. **Function Injection**: An `instruction_prompt` tells the model when and how to wrap code calls in a ````tool_code```` block.
3. **Tool Extraction**: The `extract_tool_call` function:
   - Searches for a ````tool_code```` fenced block.
   - Safely `eval()`s approved helper functions (e.g., `get_current_time`).
   - Captures and returns execution output as a ````tool_output```` block.
4. **Second Pass**: If a tool call occurred, the script appends the output to the conversation and streams a final model response.

## Extending the Toolbox

- Add new safe functions to `extract_tool_call` by:
  1. Defining the function in `main.py`.
  2. Adding it to the `safe_globals` dict.
  3. Updating the `instruction_prompt` so the model knows it’s available.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

1. Fork the repo.
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes and push: `git push origin feature/your-feature`
4. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

