"""Metadata extraction from generated images.

Supports ComfyUI, Automatic1111, and NovelAI metadata formats.
"""

import re
import json
from typing import Any, Optional, IO
from pathlib import Path
from PIL import Image, PngImagePlugin


class MetadataError(Exception):
    """Raised when metadata cannot be parsed or is invalid."""
    pass


class MetadataReader:
    """Reads and parses metadata from generated images.
    """

    def __init__(self, image_info: dict[str, Any]):
        """Initialize with raw image metadata dictionary."""
        self.image_info = image_info

    @classmethod
    def from_file(cls, path: Path | str) -> "MetadataReader":
        """Load metadata from an image file path."""
        path = Path(path)
        try:
            img = Image.open(path)
            return cls.from_image(img)
        except Exception as e:
            raise MetadataError(f"Failed to read metadata from {path}: {e}")

    @classmethod
    def from_image(cls, img: Image.Image) -> "MetadataReader":
        """Load metadata from a PIL Image."""
        _ = img.load()
        if not img.info:
            raise MetadataError("Image has no metadata")
        return cls(img.info) #type: ignore

    @classmethod
    def from_stream(cls, data: IO[bytes]) -> "MetadataReader":
        """Load metadata from a file-like object."""
        try:
            img = Image.open(data)
            return cls.from_image(img)
        except Exception as e:
            raise MetadataError(f"Failed to read metadata from stream: {e}")

    def has_comfy_metadata(self) -> bool:
        return 'prompt' in self.image_info and 'workflow' in self.image_info

    def has_a1111_metadata(self) -> bool:
        if 'parameters' not in self.image_info:
            return False
        params = self.image_info['parameters']
        return isinstance(params, (str, PngImagePlugin.iTXt))

    def has_nai_metadata(self) -> bool:
        return 'Comment' in self.image_info and 'Description' in self.image_info

    def read_comfy(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Parse ComfyUI metadata."""
        if not self.has_comfy_metadata():
            raise MetadataError("No ComfyUI metadata found (missing 'prompt' or 'workflow' fields)")
        try:
            prompt = json.loads(self.image_info['prompt'])
            workflow = json.loads(self.image_info['workflow'])
            return prompt, workflow
        except (json.JSONDecodeError, KeyError) as e:
            raise MetadataError(f"Invalid ComfyUI metadata format: {e}")

    def read_comfy_prompt(self) -> dict[str, Any]:
        """Parse ComfyUI prompt only."""
        prompt, _ = self.read_comfy()
        return prompt

    def read_a1111(self) -> dict[str, Any]:
        """Parse Automatic1111 metadata.
        Returns:
            Dictionary with keys:
            - positive: Positive prompt text
            - negative: Negative prompt text
            - gen_params: Generation parameters dict
        """
        if not self.has_a1111_metadata():
            raise MetadataError("No A1111 metadata found (missing 'parameters' field)")

        parameters = str(self.image_info['parameters'])
        return self._parse_a1111_string(parameters)

    def read_nai(self) -> dict[str, Any]:
        """Parse NovelAI metadata. Returns same structure as A1111."""
        if not self.has_nai_metadata():
            raise MetadataError("No NovelAI metadata found (missing 'Comment' or 'Description' fields)")

        try:
            gen_params = json.loads(self.image_info['Comment'])
            gen_params['Model'] = 'NovelAI'
            negative = gen_params.pop('uc', '')
            positive = self.image_info['Description']

            return {
                'positive': self._nai_to_webui(positive),
                'negative': self._nai_to_webui(negative),
                'gen_params': gen_params
            }
        except (json.JSONDecodeError, KeyError) as e:
            raise MetadataError(f"Invalid NovelAI metadata format: {e}")

    def get_model_hash(self) -> Optional[str]:
        """Extract model hash from A1111 metadata."""
        try:
            metadata = self.read_a1111()
            return metadata.get('gen_params', {}).get('Model hash')
        except MetadataError:
            return None

    @staticmethod
    def _nai_to_webui(prompt: str) -> str:
        """Convert NovelAI bracket syntax to WebUI format.

        NAI uses {} for emphasis, WebUI uses (). This is approximate
        since NAI strength is 1.05 per bracket vs 1.1 for WebUI.
        """
        replacements = {'{': '(', '}': ')'}
        return ''.join(replacements.get(char, char) for char in prompt)

    @staticmethod
    def _parse_a1111_string(parameters: str) -> dict[str, Any]:
        """Parse A1111 parameters string into structured data."""
        # Pattern for prompt with negative
        pattern_with_neg = re.compile(
            r"([\S\s]*)\nNegative prompt: ([\s\S]*)\n(Steps: [\s\S]*)"
        )
        # Pattern for prompt without negative
        pattern_without_neg = re.compile(r"([\s\S]*)\n(Steps: [\s\S]*)")

        # Try to match with negative prompt
        if "Negative prompt:" in parameters:
            match = pattern_with_neg.fullmatch(parameters)
            if not match:
                raise MetadataError(f"Invalid A1111 metadata format with negative prompt")
            positive, negative, gen_params_str = match.groups()
        else:
            # Try without negative prompt
            match = pattern_without_neg.fullmatch(parameters)
            if not match:
                raise MetadataError(f"Invalid A1111 metadata format without negative prompt")
            positive, gen_params_str = match.groups()
            negative = ""

        # Parse generation parameters
        # Remove prompt templates if present
        if "Template:" in gen_params_str:
            gen_params_str = gen_params_str[:gen_params_str.find("Template:")]

        gen_params = {}
        for param in gen_params_str.split(", "):
            parts = param.split(": ", 1)
            if len(parts) == 2:
                gen_params[parts[0]] = parts[1]

        return {
            'positive': positive,
            'negative': negative,
            'gen_params': gen_params
        }


def read_comfy_metadata(filename: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    reader = MetadataReader.from_file(filename)
    return reader.read_comfy()

def comfy_metadata_from_stream(stream: IO[bytes]) -> tuple[dict[str, Any], dict[str, Any]]:
    reader = MetadataReader.from_stream(stream)
    return reader.read_comfy()

def a1111_metadata_from_file(filename: Path) -> dict[str, Any]:
    reader = MetadataReader.from_file(filename)
    return reader.read_a1111()