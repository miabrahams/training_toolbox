import os
import json
from pathlib import Path
from typing import Tuple, Optional
from google import genai
from google.genai import types
from tqdm import tqdm
from logging import getLogger

from lib.config import get_settings, get_path

logger = getLogger(__name__)

MODEL = "gemini-1.5-flash-latest" # Updated to a common Gemini model, adjust if needed

settings = get_settings()
CAPTIONER_CFG = settings.get("captioner", {})
INPUT_DIR = CAPTIONER_CFG.get("input_dir", "./data/input")
OUTPUT_DIR = CAPTIONER_CFG.get("output_dir", "./data/output")
ERROR_DIR = CAPTIONER_CFG.get("error_dir", "./data/errors")


class CaptionProcessor:
    def __init__(self, input_dir: str, output_dir: str, error_dir: str, api_key: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.error_dir = Path(error_dir)

        # Create output directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.error_dir.mkdir(parents=True, exist_ok=True)

        self.client = genai.Client(api_key=api_key)

        self.generation_config = types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=300
        )

        self.system_prompt = """You are a caption processor. Your task for this image caption is:
1. If the caption is vague, extremely long and repetitive, or uses broken english, respond with exactly: "ERROR: FLAGGED {removal reason}" filling in the appropriate reason.
2. Otherwise, return the caption as-is, but without any description of artist logos or signatures. This usually means omitting a single sentence at the end.
3. Do not add explanations or extra text to your response."""

    def process_caption(self, caption_text: str) -> Tuple[str, Optional[str]]:
        """
        Process a single caption through the LLM.
        Returns: (processed_text, is_error)
        """
        try:

            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    self.system_prompt,
                    caption_text
                ],
                config=self.generation_config
            )

            result = response.text
            if result is None:
                raise ValueError("No response text from LLM", result)

            # Check if LLM flagged it as error
            if result.startswith("ERROR: FLAGGED"):
                return caption_text, result

            return result, None

        except Exception as e:
            print(f"API Error: {e}")
            return caption_text, f"API Error: {e}"

    def process_all_captions(self):
        """Process all caption files in the input directory."""
        # Get all text files
        caption_files = list(self.input_dir.glob("*.txt"))

        if not caption_files:
            print(f"No .txt files found in {self.input_dir}")
            return

        print(f"Found {len(caption_files)} caption files to process")

        processed_count = 0
        error_count = 0

        for file_path in tqdm(caption_files, desc="Processing captions"):
            try:
                # Read caption file
                with open(file_path, 'r', encoding='utf-8') as f:
                    caption_text = f.read().strip()

                if not caption_text:
                    continue

                # Process through LLM
                processed_text, err_msg = self.process_caption(caption_text)

                # Determine output directory and save
                if err_msg is not None:
                    logger.error(f"Flagged as error: {err_msg}\nFile: {file_path}")
                    output_path = self.error_dir / file_path.name
                    error_count += 1
                else:
                    output_path = self.output_dir / file_path.name
                    processed_count += 1

                # Write processed caption
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(processed_text)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                error_count += 1

        print("\nProcessing complete!")
        print(f"Successfully processed: {processed_count}")
        print(f"Flagged as errors: {error_count}")

def main():

    # Get API key from secrets or environment variable
    api_key = get_path("captioner.api_key") or os.getenv("CAPTION_API_KEY")
    if not api_key:
        print("Please set CAPTION_API_KEY environment variable with your Google API Key.")
        return

    # Process captions
    processor = CaptionProcessor(INPUT_DIR, OUTPUT_DIR, ERROR_DIR, api_key)
    processor.process_all_captions()

if __name__ == "__main__":
    main()
