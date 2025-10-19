
import os
import pprint
import json
from typing import Any, Dict, Union
from copy import deepcopy
import sqlite3


from lib.comfy_schemas.comfy_analysis import ComfyImage, fileToComfyImage, SchemaNode, Schema3, extract_positive_prompt
# from lib.comfy_analysis import Schema2, get_parent, prompt_schedule_gen05, prompt_schedule_gen04

pp = pprint.PrettyPrinter()
# Test single image

# TODO: Use sqlite dump
data_path = 'data/'

images = [img for img in os.listdir(data_path) if img[-4:] == '.png']

print(f"Processing {len(images)} images")
ImageObjects = [fileToComfyImage(data_path + img) for img in images]

img = [img for img in ImageObjects if 'Dragon' in img.filename][0]