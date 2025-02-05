from urllib.request import urlopen
import os
import json
import requests
import pandas as pd
from tqdm import tqdm
import re

import requests
import PIL
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import tempfile


BASE_PATH = "E:/AI/Training/PromptRating"
LOGS_PATH = f"{BASE_PATH}/DCE_Test"
DATA_PATH = f"{BASE_PATH}/Data"
TEMP_FILE = f"{DATA_PATH}/temp.png"
DATA_FILE = f"{DATA_PATH}/raw_messages.pkl"



# Read downloaded log files
filenames = os.listdir(LOGS_PATH)
filename = filenames[0]
with open(f"{LOGS_PATH}/{filename}", "r", encoding="utf8") as f:
    dataset_raw = json.load(f)


# Extract useful info from JSON
dataset = []
failed_messages = []
for i, message in enumerate(dataset_raw['messages']):
    try:
        # Simple data fields
        message_clean = {}
        message_clean['id'] = int(message['id'])
        message_clean['channel'] = message['content'].split("|")[1].strip()
        message_clean['timestamp'] = message['timestamp']
        message_clean['star_count'] = int(message['content'].split("**")[1])
        # Reactions to the #starboard message (not original)
        reactions = [int(r['count']) for r in message['reactions']]
        message_clean['reactions_top'] = max(reactions)
        message_clean['reactions_sum'] = sum(reactions)
        # Image data
        message_clean['author'] = message['embeds'][0]['author']['name']
        message_clean['image_url'] = message['embeds'][0]['image']['url']
        message_clean['image_width'] = message['embeds'][0]['image']['width']
        message_clean['image_height'] = message['embeds'][0]['image']['height']
        dataset.append(message_clean)
    except KeyError as e:
        print(f"Could not parse message {i}")
        print(f"Error: {e}")
        print("Message:")
        print(message)
        failed_messages.append(message)





# Convert to Pandas
df = pd.DataFrame.from_dict(dataset)
for url in df['image_url']: print(url)


# Get tags for each image (Makes web requests)
metadata = []
for png_url in tqdm(df['image_url']):
    # Write to disk
    with open(TEMP_FILE, "wb") as fd:
        # Download partial contents
        response = requests.get(png_url, stream=True)
        content = response.iter_content(chunk_size=128)
        for iteration in range(100):
            fd.write(next(content))
        response.close()
    im = Image.open(TEMP_FILE)
    im.load()
    metadata.append(im.info)


# Save to disk
df['metadata'] = pd.Series(metadata)
df.to_pickle(DATA_FILE)






###########################################
################# Load ####################
###########################################

df = pd.read_pickle(DATA_FILE)

df

df['metadata']
image_properties = ['exif', 'dpi', 'srgb', 'icc_profile', 'photoshop']

tags = []
stash = None
pattern_with_negative = re.compile("^([\s\S]*)\nNegative prompt: ([\s\S]*)\n(Steps: .*)$")
pattern_without_negative = re.compile("^([\s\S]*)\n(Steps: .*)$")
for meta_item in df['metadata']:
    tags_item = {}
    match meta_item:
        case {'Software': software}:
            # NovelAI and Photoshop
            if software == 'NovelAI':
                gen_params = json.loads(meta_item['Comment'])
                gen_params['Model'] = 'NovelAI'
                negative = gen_params.pop('uc')
                positive = meta_item['Description']
                tags_item = {'positive': positive,
                             'negative': negative,
                             'gen_params': gen_params}
            elif software[0:5] == 'Adobe':
                # Does not save metadata
                pass
            else:
                print("Unknown software in metadata:")
                print(meta_item)
                print("\n")
        case {'parameters': parameters} if type(parameters) in [str, PIL.PngImagePlugin.iTXt]:
            # Stable Diffusion: all as one string
            if parameters.find("Negative prompt:") > -1:
                match = pattern_with_negative.match(parameters)
                [positive, negative, gen_params] = match.groups()
            else:
                match = pattern_without_negative.match(parameters)
                [positive, gen_params] = match.groups()
                negative = ""
            # Convert generation params into dictionary
            gen_params = gen_params.split(", ")
            gen_params = [param.split(": ") for param in gen_params]
            gen_params = {param[0]: param[1] for param in gen_params}
            tags_item = {'positive': positive,
                        'negative': negative,
                        'gen_params': gen_params}
        case {'parameters': parameters}:
            # Stable diffusion: dictionary of values
            print("Complex parameters")
            print(parameters)
            print("\n")
            tags_item = parameters
        case _ if not meta_item:
            # print("No metadata.")
            pass
        case _ if all([key in image_properties for key in meta_item.keys()]):
            # print("Only EXIF data.")
            pass
        case _:
            print("Unknown metadata format.")
            print(meta_item)
            print("\n")
    tags.append(tags_item)


tags

string_tags = [t for t in tags if type(t) == str]
string_tags

