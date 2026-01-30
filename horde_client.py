"""
Horde API Client for AI Horde text generation services.

This client provides a Python interface to the AI Horde API, supporting
both synchronous and asynchronous text generation with proper error handling
and OpenAI-compatible response formatting.
"""

import json
import time
import uuid
from typing import Dict, List, Optional, Union, AsyncGenerator, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class HordeClient:
    """
    Client for interacting with the AI Horde API.

    Provides methods for model discovery, text generation submission,
    status checking, and streaming responses.
    """

    def __init__(self, api_key: str, base_url: str = "https://aihorde.net/api/v2"):
        """
        Initialize the Horde client.

        Args:
            api_key: Your AI Horde API key
            base_url: Base URL for the AI Horde API (defaults to official horde)
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

        # Configure session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set default headers
        self.session.headers.update(
            {
                "Apikey": self.api_key,
                "Content-Type": "application/json",
                "User-Agent": "HordeClient/1.0",
                "Client-Agent": "CodeHorde:1.0",
                "Accept": "application/json",
            }
        )

    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """
        Make HTTP request to the API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            **kwargs: Additional arguments for requests

        Returns:
            requests.Response object

        Raises:
            requests.RequestException: If request fails
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response

    def get_workers(self) -> List[Dict[str, Any]]:
        """
        Get list of available text generation workers from the horde.

        Returns:
            List of worker dictionaries containing worker information

        Raises:
            requests.RequestException: If request fails
        """
        response = self._make_request("GET", "workers?type=text")
        return response.json()

    def get_model_status(self) -> List[Dict[str, Any]]:
        """
        Get model statistics from the horde.

        Returns:
            List of model dictionaries containing model statistics

        Raises:
            requests.RequestException: If request fails
        """
        response = self._make_request("GET", "status/models?type=text")
        return response.json()

    def get_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available text and coding models from the horde.

        Returns:
            List of model dictionaries containing model information with enhanced metadata

        Raises:
            requests.RequestException: If request fails
        """
        workers = self.get_workers()
        model_status = self.get_model_status()

        models_dict = {}

        for model_stat in model_status:
            model_name = model_stat["name"]
            models_dict[model_name] = {
                "name": model_name,
                "count": model_stat.get("count", 0),
                "performance": model_stat.get("performance", 0),
                "queued": model_stat.get("queued", 0),
                "eta": model_stat.get("eta", 0),
                "workers": set(),
                "models_list": [],
            }

        for worker in workers:
            if worker.get("type") == "text" and worker.get("online"):
                worker_models = worker.get("models", [])
                for model_name in worker_models:
                    if model_name not in models_dict:
                        models_dict[model_name] = {
                            "name": model_name,
                            "count": 0,
                            "performance": 0,
                            "queued": 0,
                            "eta": 0,
                            "workers": set(),
                            "models_list": [],
                        }
                    models_dict[model_name]["workers"].add(worker["name"])
                    models_dict[model_name]["models_list"].append(
                        {
                            "worker_name": worker["name"],
                            "worker_id": worker["id"],
                            "worker_performance": worker.get("performance"),
                            "threads": worker.get("threads", 1),
                            "trusted": worker.get("trusted", False),
                        }
                    )

        result = []
        for model_name, model_data in models_dict.items():
            context_length = self._extract_context_limit(model_name, None)
            max_output_tokens = self._extract_output_limit(model_name, None)

            model_info = {
                "name": model_name,
                "type": "text",
                "performance": {
                    "speed": model_data["performance"],
                    "context_length": context_length,
                    "max_length": max_output_tokens,
                },
                "queued": model_data["queued"],
                "jobs": 0,
                "eta": model_data["eta"],
                "workers": len(model_data["workers"]),
                "context_length": context_length,
                "max_output_tokens": max_output_tokens,
                "speed": model_data["performance"],
                "estimated_speed": model_data["performance"],
                "count": model_data["count"],
                "worker_details": model_data["models_list"],
                "worker_names": list(model_data["workers"]),
            }

            result.append(model_info)

        return result

    def _extract_context_limit(
        self, model_name: str, context_length: Optional[int]
    ) -> int:
        """
        Extract context limit with intelligent fallbacks based on model patterns.

        Args:
            model_name: Name of the model
            context_length: Context length from performance data

        Returns:
            Context limit in tokens
        """
        if context_length and isinstance(context_length, int) and context_length > 0:
            return context_length

        # Fallback based on model name patterns
        model_name_lower = model_name.lower()

        # Common model context size patterns
        if any(x in model_name_lower for x in ["32k", "32b"]):
            return 32768
        elif any(x in model_name_lower for x in ["8k", "8b"]):
            return 8192
        elif any(x in model_name_lower for x in ["4k", "4b"]):
            return 4096
        elif any(x in model_name_lower for x in ["16k", "16b"]):
            return 16384
        elif any(x in model_name_lower for x in ["64k", "64b"]):
            return 65536
        elif any(x in model_name_lower for x in ["128k", "128b"]):
            return 128000
        elif "llama" in model_name_lower:
            # LLaMA models typically have these context sizes
            if "70b" in model_name_lower or "65b" in model_name_lower:
                return 4096  # Older LLaMA models
            else:
                return 4096  # Default for most LLaMA models
        elif "mixtral" in model_name_lower:
            return 32768  # Mixtral typically has 32k context
        elif "mistral" in model_name_lower:
            return 32768  # Mistral models typically have 32k context
        else:
            return 4096  # Conservative default

    def _extract_output_limit(self, model_name: str, max_length: Optional[int]) -> int:
        """
        Extract output limit with intelligent fallbacks based on model patterns.

        Args:
            model_name: Name of the model
            max_length: Max output length from performance data

        Returns:
            Maximum output tokens
        """
        if max_length and isinstance(max_length, int) and max_length > 0:
            return max_length

        # Fallback based on model name patterns
        model_name_lower = model_name.lower()

        # Common output size patterns (usually much smaller than context)
        if any(x in model_name_lower for x in ["32k", "32b"]):
            return 8192  # Conservative 25% of context
        elif any(x in model_name_lower for x in ["8k", "8b"]):
            return 2048
        elif any(x in model_name_lower for x in ["4k", "4b"]):
            return 1024
        elif any(x in model_name_lower for x in ["16k", "16b"]):
            return 4096
        elif any(x in model_name_lower for x in ["64k", "64b"]):
            return 16384
        elif any(x in model_name_lower for x in ["128k", "128b"]):
            return 32768
        elif "mixtral" in model_name_lower:
            return 8192  # Mixtral can generate longer outputs
        elif "mistral" in model_name_lower:
            return 4096
        else:
            return 2048  # Conservative default for most models

    def submit_generation(
        self,
        prompt: str,
        models: Optional[List[str]] = None,
        n: int = 1,
        max_output_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Submit an asynchronous text generation request.

        Args:
            prompt: The text prompt to generate from
            models: List of specific model names to use (None = auto-select)
            n: Number of generations to create
            max_output_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling parameter (0.0-1.0)
            repetition_penalty: Repetition penalty factor
            stop_sequences: List of stop sequences
            **kwargs: Additional generation parameters

        Returns:
            Dictionary containing generation job ID and other metadata

        Raises:
            requests.RequestException: If request fails
            ValueError: If parameters are invalid
        """
        # Validate parameters
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if n < 1:
            raise ValueError("n must be at least 1")

        if not (0.0 <= temperature <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")

        if not (0.0 <= top_p <= 1.0):
            raise ValueError("top_p must be between 0.0 and 1.0")

        # Build request payload
        payload = {
            "prompt": prompt,
            "n": n,
            "params": {
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
            },
        }

        if models:
            payload["models"] = models

        if max_output_tokens:
            payload["params"]["max_output_tokens"] = max_output_tokens

        if stop_sequences:
            payload["params"]["stop_sequences"] = stop_sequences

        # Add any additional parameters
        payload["params"].update(kwargs)

        response = self._make_request("POST", "generate/text/async", json=payload)
        return response.json()

    def get_generation_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a generation job.

        Args:
            job_id: The ID of the generation job

        Returns:
            Dictionary containing job status and results if available

        Raises:
            requests.RequestException: If request fails
            ValueError: If job_id is invalid
        """
        if not job_id.strip():
            raise ValueError("Job ID cannot be empty")

        response = self._make_request("GET", f"generate/text/status/{job_id}")
        return response.json()

    def wait_for_generation(
        self, job_id: str, timeout: int = 300, poll_interval: int = 2
    ) -> Dict[str, Any]:
        """
        Wait for a generation job to complete.

        Args:
            job_id: The ID of the generation job
            timeout: Maximum time to wait in seconds
            poll_interval: Time between status checks in seconds

        Returns:
            Dictionary containing completed generation results

        Raises:
            requests.RequestException: If request fails
            TimeoutError: If job doesn't complete within timeout
            ValueError: If job_id is invalid
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.get_generation_status(job_id)

            if status.get("is_possible") is False:
                raise RuntimeError(
                    f"Generation job {job_id} failed: {status.get('message', 'Unknown error')}"
                )

            if status.get("finished", 0) >= status.get("request", {}).get("n", 1):
                return status

            time.sleep(poll_interval)

        raise TimeoutError(
            f"Generation job {job_id} did not complete within {timeout} seconds"
        )

    def convert_to_openai_format(
        self, horde_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert Horde API response to OpenAI-compatible format.

        Args:
            horde_response: Response from Horde API

        Returns:
            OpenAI-formatted response dictionary
        """
        if "generations" not in horde_response:
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "horde-default",
                "choices": [],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }

        choices = []
        total_completion_tokens = 0

        for i, gen in enumerate(horde_response["generations"]):
            if "text" in gen:
                choices.append(
                    {
                        "index": i,
                        "message": {"role": "assistant", "content": gen["text"]},
                        "finish_reason": gen.get("finish_reason", "stop"),
                    }
                )

                # Estimate token count (rough approximation)
                total_completion_tokens += len(gen["text"].split())

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": horde_response.get(
                "name", horde_response.get("model", "horde-default")
            ),
            "choices": choices,
            "usage": {
                "prompt_tokens": len(horde_response.get("prompt", "").split()),
                "completion_tokens": total_completion_tokens,
                "total_tokens": len(horde_response.get("prompt", "").split())
                + total_completion_tokens,
            },
        }

    def close(self):
        """Close the HTTP session."""
        self.session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
