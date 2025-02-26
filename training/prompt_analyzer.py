#!/usr/bin/env python3
"""
Discord Prompt Analyzer

Analyzes image prompts and metadata from Discord logs exported with DiscordChatExporter.
"""

import os
import json
import requests
import pandas as pd
from tqdm import tqdm
import argparse
from PIL import Image, ImageFile

import sys
sys.path.append('..')
from lib.metadata import tags_from_metadata
from lib.prompt_parser import parse_prompt_attention

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class PromptAnalyzer:
    def __init__(self, base_path=None):
        """Initialize the PromptAnalyzer with configurable paths."""
        if base_path is None:
            # Use a default path if none provided
            base_path = os.path.join(os.path.expanduser("~"), "prompt_rating")

        # Setup paths
        self.base_path = base_path
        self.logs_path = os.path.join(base_path, "discord_logs")
        self.data_path = os.path.join(base_path, "data")
        self.temp_file = os.path.join(self.data_path, "temp.png")
        self.data_file = os.path.join(self.data_path, "processed_messages.pkl")

        # Create directories if they don't exist
        os.makedirs(self.logs_path, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)

    def load_discord_data(self, filename=None):
        """Load and parse Discord chat export data."""
        if filename is None:
            # Use the first file in the logs directory
            filenames = os.listdir(self.logs_path)
            if not filenames:
                raise FileNotFoundError(f"No log files found in {self.logs_path}")
            filename = filenames[0]

        filepath = os.path.join(self.logs_path, filename)
        print(f"Loading data from {filepath}")

        with open(filepath, "r", encoding="utf8") as f:
            dataset_raw = json.load(f)

        return self._extract_message_data(dataset_raw)

    def _extract_message_data(self, dataset_raw):
        """Extract relevant information from raw Discord data."""
        dataset = {}
        failed_messages = []

        for i, message in enumerate(dataset_raw['messages']):
            try:
                # Extract basic message data
                message_clean = {
                    'id': int(message['id']),
                    'channel': message['content'].split("|")[1].strip(),
                    'timestamp': message['timestamp'],
                    'star_count': int(message['content'].split("**")[1])
                }

                # Get reaction data
                reactions = [int(r['count']) for r in message['reactions']]
                message_clean['reactions_top'] = max(reactions) if reactions else 0
                message_clean['reactions_sum'] = sum(reactions)

                # Get image data
                message_clean['author'] = message['embeds'][0]['author']['name']
                message_clean['image_url'] = message['embeds'][0]['image']['url']
                message_clean['image_width'] = message['embeds'][0]['image']['width']
                message_clean['image_height'] = message['embeds'][0]['image']['height']

                dataset['id'] = message_clean
            except Exception as e:
                print(f"Could not parse message {i}")
                print(f"Error: {e}")
                failed_messages.append(message)

        print(f"Successfully parsed {len(dataset)} messages")
        print(f"Failed to parse {len(failed_messages)} messages")

        return pd.DataFrame.from_dict(dataset), failed_messages

    def extract_image_metadata(self, df):
        """Download images and extract their metadata."""
        metadata = []

        print("Downloading image metadata...")
        for png_url in tqdm(df['image_url']):
            try:
                # Download and write enough of the image to get metadata
                with open(self.temp_file, "wb") as fd:
                    response = requests.get(png_url, stream=True)
                    content = response.iter_content(chunk_size=128)
                    # Only read the beginning of the file to get metadata
                    for _ in range(100):
                        fd.write(next(content))
                    response.close()

                im = Image.open(self.temp_file)
                im.load()  # Required to access image info
                metadata.append(im.info)
            except Exception as e:
                print(f"Error processing {png_url}: {e}")
                metadata.append({})

        df['metadata'] = pd.Series(metadata)
        return df

    def process_data(self, filename=None, save=True):
        """Main processing function."""
        # Load and process Discord data
        df, failed = self.load_discord_data(filename)

        # Extract metadata from images
        df = self.extract_image_metadata(df)

        # Extract tags from metadata
        tags_list = tags_from_metadata(df['metadata'].tolist())
        df['tags'] = pd.Series(tags_list)

        # Extract prompts
        df['positive_prompt'] = df['tags'].apply(lambda x: x.get('positive', '') if isinstance(x, dict) else '')
        df['negative_prompt'] = df['tags'].apply(lambda x: x.get('negative', '') if isinstance(x, dict) else '')

        # Save processed data
        if save:
            os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
            df.to_pickle(self.data_file)
            print(f"Data saved to {self.data_file}")

        return df

    def analyze_prompts(self, df=None):
        """Analyze prompt data."""
        if df is None:
            # Load previously processed data
            if not os.path.exists(self.data_file):
                print(f"No processed data found at {self.data_file}")
                return None
            df = pd.read_pickle(self.data_file)

        # Get prompt statistics
        print("\n=== Prompt Analysis ===\n")

        # Count images with prompt data
        with_prompts = df['positive_prompt'].str.len() > 0
        print(f"Images with prompt data: {with_prompts.sum()} of {len(df)} ({with_prompts.mean():.1%})")

        # Display some sample prompts with their ratings
        print("\n=== Sample Prompts ===\n")
        for i, row in df[with_prompts].sort_values('star_count', ascending=False).head(5).iterrows():
            print(f"Stars: {row['star_count']}")
            print(f"Positive: {row['positive_prompt'][:100]}...")
            if row['negative_prompt']:
                print(f"Negative: {row['negative_prompt'][:100]}...")
            print()

        # Parse some prompts to demonstrate the parser
        print("\n=== Parsed Prompts ===\n")
        for prompt in df[with_prompts]['positive_prompt'].head(3):
            # Clean up prompt
            cleaned = prompt.replace('\n', ", ")
            parsed = parse_prompt_attention(cleaned)
            print(f'Raw: {cleaned[:50]}...')
            print(f'Parsed: {parsed[:3]}...')
            print()

        return df

def main():
    """Main entry point for script."""
    parser = argparse.ArgumentParser(description='Analyze Discord image prompts')
    parser.add_argument('-p', '--path', help='Base path for data', default=None)
    parser.add_argument('-f', '--file', help='Specific log file to analyze', default=None)
    args = parser.parse_args()

    # Initialize analyzer
    analyzer = PromptAnalyzer(base_path=args.path)

    # Process the data
    df = analyzer.process_data(filename=args.file)

    # Analyze the prompts
    analyzer.analyze_prompts(df)

if __name__ == "__main__":
    main()
