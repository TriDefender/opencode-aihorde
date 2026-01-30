#!/usr/bin/env python3
"""
LCPP-compatible Server for Horde AI
OpenAI-compatible API server that translates requests to/from Horde AI
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional
import aiohttp
from aiohttp import web, ClientSession, ClientTimeout
from aiohttp.web import middleware, Request, Response
import aiohttp_cors


class HordeClient:
    """Client for interacting with Horde AI API"""

    def __init__(self, api_key: str, base_url: str = "https://aihorde.net/api/v2"):
        self.api_key = api_key
        self.base_url = base_url
        self.session: Optional[ClientSession] = None

    async def __aenter__(self):
        self.session = ClientSession(
            headers={"apikey": self.api_key}, timeout=ClientTimeout(total=300)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_models(self) -> List[Dict[str, Any]]:
        """Get available models from Horde"""
        if not self.session:
            raise RuntimeError("HordeClient not initialized as context manager")

        async with self.session.get(f"{self.base_url}/status/models") as resp:
            resp.raise_for_status()
            data = await resp.json()

            models = []
            for model_name, model_info in data.items():
                model_type = model_info.get("type", "").lower()
                if "text" in model_type or "coding" in model_type:
                    models.append(
                        {
                            "id": model_name,
                            "object": "model",
                            "created": 0,
                            "owned_by": "horde",
                        }
                    )

            return models

    async def generate_completion(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate completion using Horde API"""
        if not self.session:
            raise RuntimeError("HordeClient not initialized as context manager")

        payload = {
            "prompt": prompt,
            "n": 1,
            "params": {
                "temperature": kwargs.get("temperature", 1.0),
                "top_p": kwargs.get("top_p", 1.0),
                "repetition_penalty": 1.0,
            },
        }

        # Map max_tokens to max_output_tokens
        if "max_tokens" in kwargs:
            payload["params"]["max_output_tokens"] = kwargs["max_tokens"]

        # Map model to models array
        if "model" in kwargs:
            payload["models"] = [kwargs["model"]]

        # Handle stop sequences
        if "stop" in kwargs and kwargs["stop"]:
            payload["params"]["stop_sequences"] = kwargs["stop"]

        async with self.session.post(
            f"{self.base_url}/generate/text/async", json=payload
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data

    async def check_generation(self, request_id: str) -> Dict[str, Any]:
        """Check generation status"""
        if not self.session:
            raise RuntimeError("HordeClient not initialized as context manager")

        async with self.session.get(
            f"{self.base_url}/generate/text/status/{request_id}"
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data


class LCPPServer:
    """OpenAI-compatible server for Horde AI"""

    def __init__(self, api_key: str, host: str = "0.0.0.0", port: int = 8080):
        self.api_key = api_key
        self.host = host
        self.port = port
        self.app = web.Application()
        self.setup_routes()
        self.setup_cors()

    def setup_cors(self):
        """Setup CORS for browser clients"""
        cors = aiohttp_cors.setup(
            self.app,
            defaults={
                "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*",
                    allow_methods="*",
                )
            },
        )

        for route in list(self.app.router.routes()):
            cors.add(route)

    def setup_routes(self):
        """Setup API routes"""
        self.app.router.add_get("/v1/models", self.get_models)
        self.app.router.add_post("/v1/chat/completions", self.chat_completions)
        self.app.router.add_get("/health", self.health_check)

    async def health_check(self, request: Request) -> Response:
        """Health check endpoint"""
        return Response(text="OK", status=200)

    def messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI messages format to prompt string"""
        prompt_parts = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(f"{role}: {content}")

        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)

    def horde_to_openai_response(
        self, horde_response: Dict[str, Any], request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert Horde response to OpenAI format"""
        generations = horde_response.get("generations", [])
        if not generations:
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request_data.get("model", "horde-default"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": ""},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }

        generation = generations[0]
        content = generation.get("text", "")

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request_data.get("model", "horde-default"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": request_data.get("prompt_tokens", 0),
                "completion_tokens": generation.get("token_count", 0),
                "total_tokens": request_data.get("prompt_tokens", 0)
                + generation.get("token_count", 0),
            },
        }

    def create_error_response(
        self, message: str, error_type: str = "invalid_request_error", code: str = None
    ) -> Response:
        """Create OpenAI-compatible error response"""
        error_data = {"error": {"message": message, "type": error_type}}
        if code:
            error_data["error"]["code"] = code

        return Response(
            text=json.dumps(error_data),
            status=400 if error_type == "invalid_request_error" else 500,
            content_type="application/json",
        )

    async def get_models(self, request: Request) -> Response:
        """Get available models in OpenAI format"""
        try:
            async with HordeClient(self.api_key) as client:
                models = await client.get_models()
                return Response(
                    text=json.dumps({"object": "list", "data": models}),
                    content_type="application/json",
                )
        except Exception as e:
            return self.create_error_response(f"Failed to fetch models: {str(e)}")

    async def chat_completions(self, request: Request) -> Response:
        try:
            request_data = await request.json()

            if "messages" not in request_data:
                return self.create_error_response("Missing 'messages' field")

            if not isinstance(request_data["messages"], list):
                return self.create_error_response("'messages' must be an array")

            prompt = self.messages_to_prompt(request_data["messages"])

            horde_params = {
                "max_tokens": request_data.get("max_tokens", 100),
                "temperature": request_data.get("temperature", 1.0),
                "top_p": request_data.get("top_p", 1.0),
                "stop": request_data.get("stop", []),
            }

            if "model" in request_data:
                horde_params["model"] = request_data["model"]

            async with HordeClient(self.api_key) as client:
                horde_response = await client.generate_completion(
                    prompt, **horde_params
                )

                openai_response = self.horde_to_openai_response(
                    horde_response, request_data
                )
                return Response(
                    text=json.dumps(openai_response), content_type="application/json"
                )

        except json.JSONDecodeError:
            return self.create_error_response("Invalid JSON in request body")
        except Exception as e:
            return self.create_error_response(
                f"Internal server error: {str(e)}", "server_error", "internal_error"
            )

    async def start(self):
        """Start the server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        print(f"LCPP Server running on http://{self.host}:{self.port}")
        print(f"Models endpoint: http://{self.host}:{self.port}/v1/models")
        print(f"Chat completions: http://{self.host}:{self.port}/v1/chat/completions")


async def get_api_key() -> str:
    """Get API key from user input"""
    import getpass

    api_key = getpass.getpass("Enter your Horde API key: ")
    if not api_key:
        raise ValueError("API key is required")
    return api_key


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="LCPP-compatible Horde AI Server")
    parser.add_argument("--api-key", help="Horde API key", default=None)
    parser.add_argument("--host", help="Host to bind to", default="0.0.0.0")
    parser.add_argument("--port", type=int, help="Port to bind to", default=8080)

    args = parser.parse_args()

    if not args.api_key:
        api_key = await get_api_key()
    else:
        api_key = args.api_key

    server = LCPPServer(api_key, args.host, args.port)
    await server.start()

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    asyncio.run(main())
