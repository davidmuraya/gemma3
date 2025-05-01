## Project Overview

### Local conversational agent

This repository contains a python application, that provides a local conversational interface to an Ollama model (`gemma3:4b`) with built-in tool-calling capabilities.

Gemma 3:4b, is a lightweight, open model built from the same research and technology that powers Gemini 2.0 models. These are Google's most advanced, portable and responsibly developed open models yet. They are designed to run fast, directly on devices — from phones and laptops to workstations.

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

- Python 3.11+
- Create an .env file that contains the URL where to get exchange rates `EXCHANGE_RATE_SITE=....oanda.com`
- Download and run [Ollama](https://ollama.com/) locally.
- Download Gemma3 Model. Run `ollama run gemma3:4b` after installing ollama.

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
- If the model responds with a `tool_code` block (e.g., to call `get_exchange_rate(currency='TZS', new_currency='USD')`), the script will execute the code and feed the output back to the model.
- Type `quit`, `exit`, or `q` to end the session.

## How It Works

1. **Conversation Loop**: The script collects messages in a `messages` list and streams responses from the Ollama API.
2. **Function Injection**: An `instruction_prompt` tells the model when and how to wrap code calls in a ````tool_code```` block.
3. **Tool Extraction**: The `extract_tool_call` function:
   - Searches for a ````tool_code```` fenced block.
   - Eval()`s approved helper functions (e.g., `get_exchange_rate(currency='TZS', new_currency='USD')`).
   - Captures and returns execution output as a ````tool_output```` block.
4. **Second Pass**: If a tool call occurred, the script appends the output to the conversation and streams a final model response.

   ### Output
   ![image](https://github.com/user-attachments/assets/c24d9b52-a404-4f47-a44b-2c16f513041f)

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

1. Fork the repo.
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes and push: `git push origin feature/your-feature`
4. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

