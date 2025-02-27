import re
import difflib
import random
from collections import Counter
from typing import List, Dict, Set, Optional, Any

def clean_prompt(prompt: str) -> str:
    """Clean and normalize a prompt text"""
    return prompt.strip() if prompt else ""

def extract_positive_prompt(prompt: dict) -> str:
    """Extract positive prompt from ComfyUI prompt format"""
    try:
        return prompt.get('positive', '')
    except (AttributeError, TypeError):
        return ''

def common_tokens(prompts: List[str], delimiter: str = ',') -> List[str]:
    """Compute the intersection of tokens across all prompts in the cluster."""
    token_lists = [set([token.strip() for token in prompt.split(delimiter) if token.strip()]) for prompt in prompts]
    common = set.intersection(*token_lists) if token_lists else set()
    return sorted(common)

def prompt_diffs(prompt: str, baseline: str) -> str:
    """
    Use difflib to return a unified diff between the baseline prompt and the given prompt.
    """
    baseline_tokens = baseline.split()
    prompt_tokens = prompt.split()
    diff = difflib.ndiff(baseline_tokens, prompt_tokens)
    # Filter out unchanged tokens for clarity
    diff_result = ' '.join(token for token in diff if token.startswith('+') or token.startswith('-'))
    return diff_result

def extract_tags_from_prompts(prompts: List[str], delimiter: str = ',') -> Counter:
    """Extract and count individual tags from a list of prompts."""
    all_tags = []
    for prompt in prompts:
        tags = [tag.strip() for tag in prompt.split(delimiter) if tag.strip()]
        all_tags.extend(tags)
    return Counter(all_tags)

def normalize_tag_diff(tag: str) -> str:
    """
    Normalize a tag difference by removing the +/- prefix and any
    additional formatting, returning just the tag content.
    """
    # Remove +/- prefix and any whitespace
    normalized = tag.strip()
    if normalized.startswith('+') or normalized.startswith('-'):
        normalized = normalized[1:].strip()
    return normalized

def extract_normalized_diffs(prompt_a: str, prompt_b: str) -> List[str]:
    """
    Extract normalized differences between two prompts.
    """
    # Split prompts into tag sets
    tags_a = set(tag.strip() for tag in prompt_a.split(',') if tag.strip())
    tags_b = set(tag.strip() for tag in prompt_b.split(',') if tag.strip())

    # Find differences in both directions
    diffs = list(tags_a - tags_b) + list(tags_b - tags_a)

    return diffs
