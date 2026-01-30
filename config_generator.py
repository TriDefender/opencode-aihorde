#!/usr/bin/env python3
"""
Config Generator for CodeHorde

This utility handles dynamic model discovery from AI Horde and generates
OpenCode-compatible configuration files with real model information including
context and output limits.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from horde_client import HordeClient

logger = logging.getLogger(__name__)


class ConfigGenerator:
    """Generates OpenCode-compatible configuration from discovered Horde models."""

    def __init__(
        self, api_key: str, horde_base_url: str = "https://aihorde.net/api/v2"
    ):
        """
        Initialize the config generator.

        Args:
            api_key: AI Horde API key
            horde_base_url: Base URL for the Horde API
        """
        self.api_key = api_key
        self.horde_base_url = horde_base_url
        self.horde_client = HordeClient(api_key=api_key, base_url=horde_base_url)

    def discover_models(self) -> List[Dict[str, Any]]:
        """
        Discover available models from AI Horde.

        Returns:
            List of model dictionaries with enhanced information
        """
        try:
            logger.info("Discovering models from AI Horde...")
            raw_models = self.horde_client.get_models()

            enhanced_models = []

            if isinstance(raw_models, dict):
                model_items = raw_models.items()
            elif isinstance(raw_models, list):
                model_items = [
                    (model.get("name", "unknown"), model) for model in raw_models
                ]
            else:
                logger.warning(f"Unexpected raw_models type: {type(raw_models)}")
                return []

            for model_name, model in model_items:
                enhanced_model = self._enhance_model_info(model)
                enhanced_models.append(enhanced_model)

            logger.info(f"Discovered {len(enhanced_models)} models from AI Horde")
            return enhanced_models

        except Exception as e:
            logger.error(f"Failed to discover models: {e}")
            return []

    def _enhance_model_info(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance model information with context/output limits.

        Args:
            model: Raw model data from Horde

        Returns:
            Enhanced model dictionary with limits
        """
        enhanced = model.copy()

        # Extract model name
        model_name = model.get("name", model.get("id", "unknown"))
        enhanced["name"] = model_name

        # Use the already extracted limits from horde_client
        # The enhanced horde_client now provides context_length and max_output_tokens
        context_limit = model.get("context_length")
        output_limit = model.get("max_output_tokens")

        # Fallback to extraction from performance data if not already extracted
        if context_limit is None or output_limit is None:
            performance = model.get("performance", {})

            if context_limit is None:
                context_limit = self._extract_context_limit(performance)
            if output_limit is None:
                output_limit = self._extract_output_limit(performance)

        # Set reasonable defaults if still not available
        if context_limit is None:
            context_limit = self._guess_context_limit(model_name)
        if output_limit is None:
            output_limit = self._guess_output_limit(model_name, context_limit)

        enhanced["context_limit"] = context_limit
        enhanced["output_limit"] = output_limit

        return enhanced

    def _extract_context_limit(self, performance: Dict[str, Any]) -> Optional[int]:
        """
        Extract context limit from performance data using performance.context_length.

        Args:
            performance: Performance data from Horde model

        Returns:
            Context limit in tokens, or None if not found
        """
        # First try the standard field from the enhanced horde_client
        if "context_length" in performance:
            return int(performance["context_length"])

        # Common field names for context limit (fallback)
        context_fields = [
            "max_context_length",
            "max_position_embeddings",
            "model_max_length",
            "context_size",
            "sequence_length",
        ]

        for field in context_fields:
            if field in performance:
                return int(performance[field])

        # Also check nested structures
        if "tokenizer" in performance:
            tokenizer = performance["tokenizer"]
            for field in context_fields:
                if field in tokenizer:
                    return int(tokenizer[field])

        return None

    def _extract_output_limit(self, performance: Dict[str, Any]) -> Optional[int]:
        """
        Extract output limit from performance data using performance.max_length.

        Args:
            performance: Performance data from Horde model

        Returns:
            Output limit in tokens, or None if not found
        """
        # First try the standard field from the enhanced horde_client
        if "max_length" in performance:
            return int(performance["max_length"])

        # Common field names for output limit (fallback)
        output_fields = [
            "max_new_tokens",
            "max_output_tokens",
            "generation_limit",
            "output_length",
        ]

        for field in output_fields:
            if field in performance:
                return int(performance[field])

        return None

    def _guess_context_limit(self, model_name: str) -> int:
        """
        Guess context limit based on model name patterns.

        Args:
            model_name: Name of the model

        Returns:
            Estimated context limit in tokens
        """
        model_name_lower = model_name.lower()

        # Pattern-based context limits
        if "32k" in model_name_lower:
            return 32768
        elif "16k" in model_name_lower:
            return 16384
        elif "8k" in model_name_lower:
            return 8192
        elif "4k" in model_name_lower:
            return 4096
        elif "128k" in model_name_lower:
            return 128000
        elif "64k" in model_name_lower:
            return 65536

        # Model family-based estimates
        if "gpt-4" in model_name_lower:
            return 8192
        elif "gpt-3.5" in model_name_lower:
            return 4096
        elif "llama-2" in model_name_lower:
            return 4096
        elif "llama-3" in model_name_lower or "llama3" in model_name_lower:
            return 8192
        elif "mistral" in model_name_lower:
            return 8192
        elif "mixtral" in model_name_lower:
            return 32768
        elif "qwen" in model_name_lower:
            return 8192
        elif "claude" in model_name_lower:
            return 100000

        # Default fallback
        return 4096

    def _guess_output_limit(self, model_name: str, context_limit: int) -> int:
        """
        Guess output limit based on model and context limit.

        Args:
            model_name: Name of the model
            context_limit: Context limit in tokens

        Returns:
            Estimated output limit in tokens
        """
        # Usually output limit is smaller than context limit
        # Common ratio is around 1/4 to 1/2 of context
        estimated = min(context_limit // 2, 4096)

        # Adjust based on model patterns
        model_name_lower = model_name.lower()
        if "long" in model_name_lower or "xl" in model_name_lower:
            estimated = min(context_limit // 2, 8192)

        return max(estimated, 512)  # Minimum 512 tokens

    def generate_opencode_config(
        self, models: List[Dict[str, Any]], base_url: str = "http://127.0.0.1:8080/v1"
    ) -> Dict[str, Any]:
        """
        Generate OpenCode-compatible configuration.

        Args:
            models: List of enhanced model dictionaries
            base_url: Base URL for the CodeHorde server

        Returns:
            OpenCode configuration dictionary
        """
        config = {
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "ai-horde-local": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "AI Horde (via CodeHorde)",
                    "options": {"baseURL": base_url},
                    "models": {},
                }
            },
        }

        models_config = config["provider"]["ai-horde-local"]["models"]

        # Add each discovered model
        for model in models:
            model_name = model.get("name", "unknown")
            context_limit = model.get("context_limit", 4096)
            output_limit = model.get("output_limit", 2048)

            # Create a safe model ID for OpenCode
            safe_model_id = (
                model_name.lower().replace(" ", "-").replace("/", "-").replace("_", "-")
            )

            models_config[safe_model_id] = {
                "name": model_name,
                "limit": {"context": context_limit, "output": output_limit},
            }

        # Also add a generic "horde" model that uses default limits
        if models:
            avg_context = sum(m.get("context_limit", 4096) for m in models) // len(
                models
            )
            avg_output = sum(m.get("output_limit", 2048) for m in models) // len(models)

            models_config["horde"] = {
                "name": "All Available Horde Models",
                "limit": {"context": avg_context, "output": avg_output},
            }

        return config

    def save_config(
        self, config: Dict[str, Any], output_path: str = "generated-opencode.json"
    ) -> str:
        """
        Save configuration to file.

        Args:
            config: Configuration dictionary
            output_path: Output file path

        Returns:
            Path to the saved file
        """
        try:
            output_file = Path(output_path)
            output_file.write_text(json.dumps(config, indent=2), encoding="utf-8")
            logger.info(f"Configuration saved to {output_file.absolute()}")
            return str(output_file.absolute())

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise

    def generate_and_save_config(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        output_path: str = "generated-opencode.json",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Complete workflow: discover models and generate/save config.

        Args:
            base_url: Base URL for the CodeHorde server
            output_path: Output file path for the config

        Returns:
            Tuple of (output_file_path, generated_config)
        """
        # Discover models
        models = self.discover_models()

        if not models:
            logger.warning("No models discovered, creating minimal config")
            models = []

        # Generate config
        config = self.generate_opencode_config(models, base_url)

        # Save config
        output_file = self.save_config(config, output_path)

        return output_file, config


def main():
    """Command line interface for the config generator."""
    import argparse
    import sys
    import os

    parser = argparse.ArgumentParser(
        description="Generate OpenCode configuration from AI Horde models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="AI Horde API key (can also be set via HORDE_API_KEY env var)",
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:8080/v1",
        help="Base URL for CodeHorde server (default: http://127.0.0.1:8080/v1)",
    )

    parser.add_argument(
        "--horde-base-url",
        type=str,
        default="https://aihorde.net/api/v2",
        help="Base URL for Horde API (default: https://aihorde.net/api/v2)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="generated-opencode.json",
        help="Output file path (default: generated-opencode.json)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set up logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Get API key
    api_key = args.api_key or os.getenv("HORDE_API_KEY")
    if not api_key:
        print(
            "Error: API key required. Use --api-key or set HORDE_API_KEY environment variable."
        )
        sys.exit(1)

    # Generate config
    try:
        generator = ConfigGenerator(api_key=api_key, horde_base_url=args.horde_base_url)
        output_file, config = generator.generate_and_save_config(
            base_url=args.base_url, output_path=args.output
        )

        print(f"✅ Configuration generated successfully!")
        print(f"📁 Output file: {output_file}")
        print(
            f"🤖 Models discovered: {len(config['provider']['ai-horde-local']['models'])}"
        )

        # Show sample of discovered models
        models = list(config["provider"]["ai-horde-local"]["models"].keys())
        if models:
            print(
                f"📋 Sample models: {', '.join(models[:5])}"
                + (f" and {len(models) - 5} more..." if len(models) > 5 else "")
            )

    except Exception as e:
        logger.error(f"Failed to generate configuration: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
