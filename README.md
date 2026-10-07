# Neo — Basic AI Agent

A local conversational CLI agent with:

- OpenAI-backed chat loop
- Optional local-model mode through an OpenAI-compatible endpoint like Ollama
- Persistent conversation memory in `chat_history.json`
- Task-focused skills you can load at startup or switch during chat
- A small local tool layer
- Simple commands for reset and save

## Setup

```powershell
cd basic-agent
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

## Local App Launcher

Use this launcher to run Neo as a local Windows app:

```powershell
basic-agent\run_local_agent.bat
```

There is also a Desktop launcher:

```powershell
run_local_agent.bat
```

For a browser conversation app with chat bubbles, microphone input, and spoken replies, use:

```powershell
basic-agent\run_web_agent.bat
```

or the Desktop launcher:

```powershell
run_web_agent.bat
```

It runs locally at:

```powershell
http://127.0.0.1:8765
```

The launcher starts the local Ollama server if it is not already running, then opens Neo. Conversation memory is saved on this device at:

```powershell
basic-agent\chat_history.json
```

The web app also shows a short reasoning summary before each assistant answer when `AGENT_REASONING_ENABLED=true`. This is a concise explanation of the approach, not hidden chain-of-thought, and spoken replies read only the final answer.

For the fastest replies, turn on `AGENT_FAST_REPLY=true` or use `/fast on` in chat. Fast reply mode trims conversation history, disables reasoning summaries, and shortens completions.

The web app includes extra skills:

- `Image`: generate an image from the current prompt. If `OPENAI_IMAGE_API_KEY` is set, it uses the configured OpenAI image model; otherwise it saves a local SVG prompt card in `generated/`.
- `Video`: create a local animated storyboard HTML file from the current prompt in `generated/`.
- `Doc`: upload `.txt`, `.md`, `.csv`, `.json`, `.html`, `.docx`, or `.pdf` files and summarize them. PDF extraction requires `pypdf`.

Set `OPENAI_API_KEY` in `.env`, then run:

```powershell
python main.py
```

Or:

```powershell
python main.py chat
```

## Run With A Local Model

For Ollama, set these values in `.env`:

```env
LLM_PROVIDER=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=llama3
OPENAI_API_KEY=ollama
```

Then start Ollama and a model:

```powershell
ollama serve
ollama pull llama3
```

Then run:

```powershell
.\.venv\Scripts\python main.py chat
```

Inside the chat, use:

- `/exit`: quit
- `/reset`: clear saved conversation memory
- `/save`: force-save the current conversation
- `/where`: show where the conversation history is stored
- `/voice on`: speak agent replies
- `/voice off`: stop speaking agent replies
- `/voices`: list installed Windows text-to-speech voices
- `/listen`: listen once through the microphone and send the recognized text
- `/mic on`: keep using microphone input for each prompt
- `/mic off`: return to typed input
- `/mic-check`: show microphone receiver diagnostics
- `/recognizers`: list installed Windows speech recognizers
- `/code-help`: show local coding commands
- `/skills`: list available skills
- `/skill <name>`: activate a skill
- `/skill off`: disable the active skill
- `/fast on`: enable fast reply mode
- `/fast off`: disable fast reply mode
- `/files [path]`: list project files
- `/read <path>`: read a project file
- `/search <query>`: search project files
- `/run <python or test command>`: run an allowed project command
- `/write <path> | <complete file content>`: write a project file

Voice output uses Windows text-to-speech. Microphone input first tries Windows speech recognition, then falls back to `SpeechRecognition` with either `PyAudio` or `sounddevice`.

## Built-in tools

- `get_time`: Returns the local timestamp.
- `list_files`: Lists files in a directory inside the project.
- `read_file`: Reads a text file inside the project.

The file tools are intentionally restricted to the project directory.

## Configuration

- `LLM_PROVIDER`: `openai` or `ollama`
- `OPENAI_API_KEY`: your API key
- `OPENAI_BASE_URL`: optional custom API base URL, required for local Ollama mode
- `OPENAI_MODEL`: defaults to `gpt-4.1-mini`
- `OPENAI_IMAGE_API_KEY`: optional real OpenAI key for image generation
- `OPENAI_IMAGE_MODEL`: image model name, defaults to `gpt-image-1`
- `OPENAI_IMAGE_SIZE`: image size, defaults to `1024x1024`
- `AGENT_NAME`: display name in chat
- `AGENT_SYSTEM_PROMPT`: optional custom personality/instructions
- `AGENT_TOOLS_ENABLED`: enable model tool-calling for providers/models that support it
- `AGENT_VOICE_ENABLED`: `true` or `false`
- `AGENT_VOICE_NAME`: optional installed Windows voice name
- `AGENT_VOICE_RATE`: from `-10` to `10`
- `AGENT_VOICE_VOLUME`: from `0` to `100`
- `AGENT_LISTEN_ENABLED`: start with microphone input enabled
- `AGENT_LISTEN_ENGINE`: `auto`, `windows`, or `speech_recognition`
- `AGENT_LISTEN_CULTURE`: recognizer culture, defaults to `en-US`
- `AGENT_LISTEN_TIMEOUT_SECONDS`: microphone listen timeout, from `2` to `30`
- `AGENT_FAST_REPLY`: enable fast reply mode for lower latency
- `AGENT_REASONING_ENABLED`: show concise reasoning summaries in the web app
- `AGENT_REASONING_EFFORT`: `low`, `medium`, or `high` for OpenAI reasoning models
- `AGENT_RESPONSE_MAX_TOKENS`: cap each model answer to keep responses quick
- `AGENT_HISTORY_MAX_MESSAGES`: number of recent non-system messages sent to the model; use `0` for full history
- `AGENT_SKILL`: optional skill to load at startup
- `AGENT_SKILLS_DIR`: directory that stores local skill markdown files, defaults to `agent_skills/`

In `ollama` mode, tool-calling is disabled for compatibility and the agent runs as a pure conversational assistant.

## Skills

Skills are markdown files in [`agent_skills/`](agent_skills/).

Each file can define:

- `name`
- `description`
- the instructions the model should follow when the skill is active

Included starter skills:

- `coding`
- `data-analysis`
- `planning`
- `statistical-analysis`
- `time-series-analysis`
- `forecasting`
- `writing`
- `machine-learning`
- `deep-learning`

You can start with a skill from the command line:

```powershell
python main.py chat --skill coding
python main.py web --skill coding
```

Or switch in the CLI with `/skills` and `/skill <name>`.

## Extend it

Add tools in [agent/tools.py](agent/tools.py) and adjust conversation behavior in [agent/runner.py](agent/runner.py).

## Development checks

```bash
python -m unittest discover -s tests -v
```

## Local data and credentials

Keep API keys in your local `.env` file. Conversation history, uploads, generated output, virtual environments, and logs are excluded from version control. Copy `.env.example` to `.env` and supply your own configuration.

This is a local development agent. Review tool permissions and generated code before using them with important files. Image prompt cards and video storyboards are fallback artifacts, not full image or video generation.
