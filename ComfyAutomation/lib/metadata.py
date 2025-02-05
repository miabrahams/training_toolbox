import re
import json
from PIL import Image, PngImagePlugin
from itertools import islice
import io
import base64

def decode_image(i):
    return Image.open(io.BytesIO(base64.b64decode(i.split(",", 1)[0])))

def chunks(lst, chunk_size):
    # This iterator stores the overall state.
    lst_it = iter(lst)
    # iter(fn, sentinel) repeatedly calls a function until it returns the sentinel value.
    # tuple(islice(it, chunk_size)) will return a tuple of length chunk_size while moving the iterator forward.
    # When we return an empty tuple, iteration is finished.
    return iter(lambda: tuple(islice(lst_it, chunk_size)), ())




artist_filename_re = re.compile("^grid-[0-9]{4}-(.*)\.png")
IMAGE_PROPERTIES = ['exif', 'dpi', 'srgb', 'icc_profile', 'gamma', 'interlace', 'chromaticity', 'photoshop', 'XML:com.adobe.xmp']

def nai_to_webui(prompt):
    # NAI strength adjustment is 1.05 per parentheses instead of 1.1
    # So this isn't an exact replacement, but good enough for now maybe
    replacements = {'{': '(', '}': ')'}
    return ''.join([replacements.get(char, char) for char in prompt])


def parse_software_metadata(meta_item, software):
    # NovelAI and Photoshop
    if software == 'NovelAI':
        gen_params = json.loads(meta_item['Comment'])
        gen_params['Model'] = 'NovelAI'
        negative = gen_params.pop('uc')
        positive = meta_item['Description']
        return {'positive': nai_to_webui(positive),
                'negative': nai_to_webui(negative),
                'gen_params': gen_params}
    elif software == "Celsys Studio Tool":
        return{}
    elif software[0:5] == 'Adobe':
        return {}
    else:
        print("Unknown software in metadata:")
        print(meta_item)
        print("\n")
        return {}


PATTERN_WITH_NEGATIVE = re.compile("([\S\s]*)\nNegative prompt: ([\s\S]*)\n(Steps: [\s\S]*)")
PATTERN_WITHOUT_NEGATIVE = re.compile("([\s\S]*)\n(Steps: [\s\S]*)")
def parse_metadata(parameters):
    try:
        # Comes as one long string
        if parameters.find("Negative prompt:") > -1:
            match = PATTERN_WITH_NEGATIVE.fullmatch(parameters)
            [positive, negative, gen_params] = match.groups()
        else:
            negative = ""
            match = PATTERN_WITHOUT_NEGATIVE.fullmatch(parameters)
            [positive, gen_params] = match.groups()
        # Convert generation params into dictionary
        # Get rid of prompt templates
        if gen_params.find("Template:") > -1:
            gen_params = gen_params[:gen_params.find("Template:")]
        gen_params = gen_params.split(", ")
        gen_params = [param.split(": ") for param in gen_params]
        gen_params = {param[0]: param[1] for param in gen_params}
        return {'positive': positive,
                'negative': negative,
                'gen_params': gen_params}
    except Exception as e:
        print("Error parsing metadata.")
        raise Exception(parameters)

def tags_from_metadata(metadata):
    tags = []
    for meta_item in metadata:
        tags_item = {}
        match meta_item:
            case _ if not meta_item:
                pass
            case {'software': software}:
                tags_item = parse_software_metadata(meta_item, software)
            case {'parameters': parameters} if type(parameters) in [str, PngImagePlugin.iTXt]:
                tags_item = parse_metadata(parameters)
            case {'parameters': parameters}:
                # Stable diffusion: dictionary of values
                print("Complex parameters")
                print(parameters)
                print("\n")
                tags_item = parameters
            case _ if all([key in IMAGE_PROPERTIES for key in meta_item.keys()]):
                # Only EXIF data.
                pass
            case _:
                print("Unknown metadata format:")
                print(meta_item)
                print("\n")
        tags.append(tags_item)
    return tags

def raw_image_info(filename: str) -> dict[str, str]:
    img = Image.open(filename)
    if filename[-4:] == ".png":
        info = img.load()
        return img.info
    # This should just return exif_data and parse elsewhere
    elif filename[-4:] == ".jpg":
        exif_data = img._getexif()
        return exif_data[37510].decode('utf-8').replace('\x00', '').replace('UNICODE', '')

# Read PIL image metadata
def info_from_file(filename: str) -> dict:
    info = raw_image_info(filename);
    return info['parameters']

def metadata_from_file(filename: str) -> dict:
    info = info_from_file(filename)
    return parse_metadata(info)

# Parse raw info and extract hash
def hash_from_metadata(metadata: dict) -> str:
    return metadata['gen_params']['Model hash']

# Extract the model hash from filename
def hash_from_file(filename):
    metadata = info_from_file(filename)
    return hash_from_metadata(metadata)

# Extract the artist name from prompt
# Method = 0: "by [artist]" is at the very end of prompt
# Method = 1: "by [artist])" inside the prompt
# Method = 2: "(by [artist]:1.25)" inside the prompt
def artist_from_file(filename, method=0):
    info = info_from_file(filename)
    prompt = parse_metadata(info)['positive']
    if method == 0:
        return prompt.split("by ")[1]
    elif method == 1:
        re.match(".*by (.*)\).*", prompt).group(1)
    elif method == 2:
        return prompt.split("(by ")[1].split(":1.25")[0]
    raise Exception("Unknown method")



# Extract the artist name from filename
def artist_from_filename(filename):
    match = artist_filename_re.match(filename)
    if match:
        return match.group(1)
    else:
        return None

# Progressively parse comfyui files

def comfy_prompt(filename: str) -> dict[str, str]:
    return json.loads(raw_image_info(filename)['prompt'])

def comfy_nodes(filename: str) -> dict[str, str]:
    return {id: value for (id, value) in comfy_prompt(filename).items()}

def comfy_unique_node_types(filename: str) -> dict[str, str]:
    node_types = [value['class_type'] for (_, value) in comfy_prompt(filename).items()]
    return sorted(set(node_types))


