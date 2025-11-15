from pathlib import Path
import re

from src.lib.config import load_settings


settings = load_settings()
captioner_cfg = settings.get("captioner", {})
collator_cfg = captioner_cfg.get("collator", {})
output_dir = Path(captioner_cfg.get("output_dir", "./data/output"))
collated_file = Path(collator_cfg.get("output_file", "./data/collated_captions.txt"))


# Combined replacements dictionary:
replacements = {
    # Merge similar "illustration" starts into one pattern.
    r"^(?:Digital|Black and white (?:digital comic|comic strip|comic)) illustration\. ": r"Multiple panels. ",
    r'^(Black and white )?[cC]omic ([a-zA-z\s]+)?by (artist )?([a-zA-Z0-9\s]+)\. ': r"Multiple panels. ",
    r"^Digital (comic|comic-style) illustration with (two|three|four|five) panels\. ": r"\2 panels. ",
    r"^Digital (illustration|pencil drawing) (split into|in) (two|three|four|five) (panels|sections|main sections)\. ": r"\3 panels. ",
    r"^Digital comic (strip |illustration )?(in|with) (two|three|four|five|six|seven) (panels|sections|main sections)( in a [a-zA-z\s]+)?\. ": r"\3 panels. ",
    r'^Digital character reference': r"Character reference",
    r'The illustration (uses|uses a|has|has a|is in) ([^.]+)\. ': r'\1. ',
}

# Create a list of similar removal patterns, then combine them with alternation.
_removal_patterns = [
    r"(?:Black and white )?[dD]igital (?:comic )?(?:illustration|drawing|painting|artwork|line drawing) of (?:a |an )?",
    r"(?:Black and white )?[dD]igital [a-zA-Z\s]+by (?:artist )?[a-zA-Z0-9_,@\-\s]+\. ",
    r"Digital [a-zA-Z]+ by [a-zA-Z0-9\s]+\.",
    r"Digital (?:comic )?(?:illustration|artwork)\. ",
    r"Digital illustration in (?:grayscale|monochrome)\. ",
    r"(?:A )?Black[-\s]and[-\s]white comic illustration\. ",
    r"Digital (?:comic )?(?:illustration|drawing)? in [a-zA-Z\s]+ tones?\. ",
    r"Digital (?:drawing|comic illustration) in (?:grayscale|a monochromatic style|a sketch style)\. ",
    r"Digital sketch of (?:a |an )?",
    r"Digital drawing\. ",
    r"Digital illustration in a comic (?:strip format|style)\. ",
    r"Digital comic strip of (?:a |an )?",
    r"Comic (?:page )?illustration in (?:a )?digital medium\. ",
    r"Digital [a-zA-Z\s]+ sequence\. ",
]
combined_removals = re.compile("|".join(_removal_patterns), re.IGNORECASE)


def postprocess(caption_text: str) -> str:
    for old, new in replacements.items():
        caption_text = re.sub(old, new, caption_text)
    caption_text = re.sub(combined_removals, '', caption_text)
    caption_text = caption_text[0].upper() + caption_text[1:]

    return caption_text


RUN = True
POSTPROCESS = bool(collator_cfg.get("postprocess", True))
if RUN:
    result = []
    for input_path in output_dir.rglob("*.txt"):
        with input_path.open('r', encoding='utf-8') as f:
            caption_text = f.read().replace('\n', ' ')
        if POSTPROCESS:
            caption_text = postprocess(caption_text)
        caption_text = caption_text.strip()

        print(f"Processing {input_path.name}: {caption_text[:50]}...")  # Print first 50 chars for brevity
        result.append(caption_text)

    collated_file.parent.mkdir(parents=True, exist_ok=True)
    with collated_file.open('w', encoding='utf-8') as f:
        f.write("\n".join(result))
