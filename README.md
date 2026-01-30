# AI Horde Client

A lightweight Python client for OpenCode, providing AI assisted coding through Horde's distributed AI network.

Auto generator of opencode configuration based on capabilities of currently available models

## Setup

Use `uv` for virtual environment and dependency management:
```bash
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

## Usage

Run the server and client:
```bash
python main.py
```
You will be prompted to put in your AI Horde key.

## Note

AI Horde currently doesn't support streaming responses. All text generation is batch-based. If OpenCode adds non-streaming support one day this would work.

## Files

- `main.py` - Client interface
- `server.py` - Local server for API calls
- `horde_client.py` - Horde API wrapper
- `config_generator.py` - Configuration setup