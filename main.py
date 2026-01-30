#!/usr/bin/env python3
"""
CodeHorde - AI Horde to Llama.cpp Protocol Translation Server

This server provides a llama.cpp-compatible (OpenAI-compatible) API endpoint
that translates requests to the AI Horde distributed computing network.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import aiohttp
from aiohttp import web
import aiohttp_cors

from horde_client import HordeClient
from config_generator import ConfigGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CodeHordeServer:
    """Main server class for the Horde-to-LCPP translation service."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        api_key: Optional[str] = None,
        horde_base_url: str = "https://aihorde.net/api/v2",
        generate_config: bool = True,
    ):
        """
        Initialize the CodeHorde server.

        Args:
            host: Host to bind the server to
            port: Port to listen on
            api_key: AI Horde API key
            horde_base_url: Base URL for the Horde API
            generate_config: Whether to generate OpenCode config on startup
        """
        self.host = host
        self.port = port
        self.api_key = api_key
        self.horde_base_url = horde_base_url
        self.generate_config = generate_config
        self.horde_client: Optional[HordeClient] = None
        self.config_generator: Optional[ConfigGenerator] = None
        self.cached_models: List[Dict[str, Any]] = []
        self.app = web.Application()

        # Setup routes and CORS
        self._setup_routes()
        self._setup_cors()

    def _setup_cors(self):
        """Configure CORS for browser clients."""
        cors = aiohttp_cors.setup(
            self.app,
            defaults={
                "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    allow_methods=["GET", "POST", "OPTIONS"],
                    allow_headers=["*", "Authorization", "Content-Type"],
                )
            },
        )
        for route in list(self.app.router.routes()):
            cors.add(route)

    def _setup_routes(self):
        """Setup API routes."""
        self.app.router.add_get("/health", self.health_check)
        self.app.router.add_get("/v1/models", self.list_models)
        self.app.router.add_post("/v1/chat/completions", self.chat_completions)

    def _ensure_api_key(self) -> str:
        """Ensure API key is available."""
        if not self.api_key:
            raise web.HTTPUnauthorized(
                body=json.dumps(
                    {
                        "error": {
                            "message": "API key not configured. Please provide --api-key or set HORDE_API_KEY environment variable.",
                            "type": "authentication_error",
                            "code": "missing_api_key",
                        }
                    }
                ),
                content_type="application/json",
            )
        return self.api_key

    def _get_horde_client(self) -> HordeClient:
        """Get or create Horde client."""
        if not self.horde_client:
            api_key = self._ensure_api_key()
            self.horde_client = HordeClient(
                api_key=api_key, base_url=self.horde_base_url
            )
        return self.horde_client

    def _get_config_generator(self) -> ConfigGenerator:
        """Get or create config generator."""
        if not self.config_generator:
            api_key = self._ensure_api_key()
            self.config_generator = ConfigGenerator(
                api_key=api_key, horde_base_url=self.horde_base_url
            )
        return self.config_generator

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert OpenAI messages format to a simple prompt string.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            Formatted prompt string
        """
        if not messages:
            return ""

        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(f"{role}: {content}")

        return "\n".join(prompt_parts) + "\nAssistant:"

    def _create_openai_model_response(
        self, models: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create OpenAI-format models response.

        Args:
            models: List of model information from Horde

        Returns:
            OpenAI-format models response
        """
        model_list = []
        for model in models:
            model_id = model.get("name", model.get("id", "unknown"))
            model_list.append(
                {
                    "id": model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "horde",
                }
            )

        return {"object": "list", "data": model_list}

    async def health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({"status": "healthy"})

    async def list_models(self, request: web.Request) -> web.Response:
        """
        List available models from Horde.

        GET /v1/models
        """
        try:
            client = self._get_horde_client()
            models = client.get_models()
            response = self._create_openai_model_response(models)
            return web.json_response(response)
        except web.HTTPUnauthorized:
            raise
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            return web.json_response(
                {
                    "error": {
                        "message": f"Failed to fetch models: {str(e)}",
                        "type": "server_error",
                        "code": "model_fetch_failed",
                    }
                },
                status=500,
            )

    async def chat_completions(self, request: web.Request) -> web.Response:
        """
        Handle chat completion requests.

        POST /v1/chat/completions
        """
        try:
            # Parse request body
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response(
                {
                    "error": {
                        "message": "Invalid JSON in request body",
                        "type": "invalid_request_error",
                        "code": "parse_error",
                    }
                },
                status=400,
            )

        # Extract parameters
        model = body.get("model", "horde-default")
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens")
        temperature = body.get("temperature", 1.0)
        top_p = body.get("top_p", 1.0)
        stream = body.get("stream", False)
        stop = body.get("stop")

        # Validate messages
        if not messages:
            return web.json_response(
                {
                    "error": {
                        "message": "At least one message is required",
                        "type": "invalid_request_error",
                        "code": "missing_messages",
                    }
                },
                status=400,
            )

        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)

        # Get Horde client
        client = self._get_horde_client()

        try:
            if stream:
                return web.json_response(
                    {
                        "error": {
                            "message": "Streaming is not supported by AI Horde",
                            "type": "invalid_request_error",
                            "code": "streaming_not_supported",
                        }
                    },
                    status=400,
                )
            else:
                return await self._handle_non_streaming(
                    client, prompt, model, max_tokens, temperature, top_p, stop
                )
        except Exception as e:
            logger.error(f"Error in chat completions: {e}")
            return web.json_response(
                {
                    "error": {
                        "message": f"Generation failed: {str(e)}",
                        "type": "server_error",
                        "code": "generation_failed",
                    }
                },
                status=500,
            )

    async def _handle_non_streaming(
        self,
        client: HordeClient,
        prompt: str,
        model: str,
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
    ) -> web.Response:
        """Handle non-streaming chat completion."""
        # Submit generation with correct parameter mapping
        result = client.submit_generation(
            prompt=prompt,
            models=[model] if model != "horde-default" else None,
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop,
        )

        job_id = result.get("id")
        if not job_id:
            return web.json_response(
                {
                    "error": {
                        "message": "No job ID returned from Horde",
                        "type": "server_error",
                        "code": "no_job_id",
                    }
                },
                status=500,
            )

        # Wait for completion
        status = client.wait_for_generation(job_id)

        # Convert to OpenAI format
        response = client.convert_to_openai_format(status)
        response["model"] = model

        return web.json_response(response)

    async def _generate_opencode_config(self):
        """Generate OpenCode configuration with discovered models."""
        if not self.generate_config:
            return

        try:
            logger.info("Generating OpenCode configuration...")
            config_generator = self._get_config_generator()
            base_url = f"http://{self.host}:{self.port}/v1"
            output_file, config = config_generator.generate_and_save_config(
                base_url=base_url, output_path="generated-opencode.json"
            )
            logger.info(f"✅ OpenCode configuration generated: {output_file}")

            self.cached_models = config_generator.discover_models()
            logger.info(
                f"📊 Cached {len(self.cached_models)} models for fast responses"
            )

        except Exception as e:
            logger.warning(f"Failed to generate OpenCode config: {e}")

    async def run(self):
        """Start the server."""
        logger.info(f"Starting CodeHorde server on {self.host}:{self.port}")

        await self._generate_opencode_config()

        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        logger.info(f"CodeHorde server running at http://{self.host}:{self.port}")
        logger.info("Available endpoints:")
        logger.info(f"  - GET  http://{self.host}:{self.port}/health")
        logger.info(f"  - GET  http://{self.host}:{self.port}/v1/models")
        logger.info(f"  - POST http://{self.host}:{self.port}/v1/chat/completions")

        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            logger.info("Shutting down server...")
        finally:
            await runner.cleanup()


def prompt_for_api_key() -> str:
    """Prompt user for Horde API key."""
    print("\n" + "=" * 60)
    print("AI Horde API Key Required")
    print("=" * 60)
    print("Please enter your AI Horde API key.")
    print("You can get a free key at: https://aihorde.net/register")
    print("=" * 60 + "\n")

    api_key = input("Enter your API key: ").strip()

    if not api_key:
        print("Error: API key cannot be empty")
        sys.exit(1)

    return api_key


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CodeHorde - AI Horde to Llama.cpp Protocol Translation Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with API key from argument
  python main.py --api-key "your-horde-api-key"

  # Start with API key from environment variable
  export HORDE_API_KEY="your-horde-api-key"
  python main.py

  # Start on custom host and port
  python main.py --host 0.0.0.0 --port 9000

For more information, see: https://github.com/yourusername/codehorde
        """,
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080)",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="AI Horde API key (optional, will prompt if not provided)",
    )

    parser.add_argument(
        "--horde-base-url",
        type=str,
        default="https://aihorde.net/api/v2",
        help="Base URL for the Horde API (default: https://aihorde.net/api/v2)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--no-generate-config",
        action="store_true",
        help="Skip automatic OpenCode configuration generation on startup",
    )

    args = parser.parse_args()

    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Get API key
    api_key = args.api_key
    if not api_key:
        # Check environment variable
        api_key = api_key or __import__("os").getenv("HORDE_API_KEY")

    if not api_key:
        # Prompt user
        api_key = prompt_for_api_key()

    # Create and run server
    server = CodeHordeServer(
        host=args.host,
        port=args.port,
        api_key=api_key,
        horde_base_url=args.horde_base_url,
        generate_config=not args.no_generate_config,
    )

    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)


if __name__ == "__main__":
    main()
