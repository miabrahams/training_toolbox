
import os
import pprint
import json
from pathlib import Path

from src.lib.config import get_settings

from src.lib.comfy_schemas.comfy_analysis import ComfyImage, fileToComfyImage, SchemaNode, Schema3, extract_positive_prompt
# from src.lib.comfy_analysis import Schema2, get_parent, prompt_schedule_gen05, prompt_schedule_gen04

pp = pprint.PrettyPrinter()
# Test single image

settings = get_settings()
data_path = Path(settings.get("tools.comfy.data_path", "data")).expanduser()

images = [img for img in os.listdir(data_path) if img[-4:] == '.png']

print(f"Processing {len(images)} images")
ImageObjects = [fileToComfyImage(data_path / img) for img in images]

img = [img for img in ImageObjects if 'Dragon' in str(img)][0]
