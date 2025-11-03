import json
from PIL import Image
import torch
from torchvision.transforms import transforms
from pathlib import Path
import concurrent.futures

MODEL_PATH = "models/eva02.pth"
TAGS_PATH = "models/tags_8041_eva02.json"

def get_images(folder: Path, subfolders: bool = True):
    if subfolders:
        return list(walk(folder))
    return [
        file
        for file in folder.iterdir()
        if file.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp", ".gif"]
    ]


def walk(folder: Path):
    for p in folder.iterdir():
        if p.is_dir():
            yield from walk(p)
            continue
        if p.suffix.lower() not in [".png", ".jpg", ".jpeg", ".webp", ".gif"]:
            continue
        yield p



def transform(image: Image.Image, device, thin: bool = False):
    """Transform an image to a tensor for model input."""
    normal_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ),
    ])

    thin_transform = transforms.Compose([
        transforms.Resize(448),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ),
    ])

    if thin:
        tensor = thin_transform(image)
    else:
        tensor = normal_transform(image)

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Transform did not return a torch.Tensor")

    # Add batch dimension and move to device
    return tensor.unsqueeze(0).to(device)


def load_image(image_path: Path, device):
    image = Image.open(image_path).convert("RGB")
    ratio = image.height / image.width
    # Call transform with the appropriate parameters
    return transform(image, device, thin=(ratio > 2.0 or ratio < 0.5))

def load_model(device):
    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()
    return model

def load_tags():
    allowed_tags = None
    with open(TAGS_PATH, "r") as file:
        tags = json.load(file)
        allowed_tags = sorted(tags)
        allowed_tags.insert(0, "placeholder0")
        allowed_tags.append("placeholder1")
        allowed_tags.append("explicit")
        allowed_tags.append("questionable")
        allowed_tags.append("safe")
        allowed_tags.append("3d")
    return allowed_tags

def main():
    # Start model loading in a separate thread
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_model = executor.submit(load_model, device)
        print("Started model load.")


    # get folder input
    folder = Path(input("folder for tagging: ").strip(' "'))
    if not folder.exists():
        print("No folder found, quitting")
        quit()

    subfolder = False
    while not subfolder:
        subfolder = input("subfolder (y/n): ")
        if subfolder.lower() not in ["y", "yes", "n", "no"]:
            subfolder = False
    subfolder = subfolder.lower() in ["y", "yes"]

    # get images in folder
    images = get_images(folder, subfolder)
    if len(images) == 0:
        print("no images found, quitting")
        quit()
    total_images = len(images)

    # get threshold
    threshold = False
    while not threshold:
        try:
            threshold = float(input("Threshold: "))
        except ValueError:
            threshold = False

    prepend = input("Prepend tags: ")
    prefix = [tag.strip(", ") for tag in prepend.split(",")]

    print(f"prefix: {prefix}")


    tags = load_tags()
    tags_set = set([tag.replace("_", " ") for tag in tags])

    # load model
    model = future_model.result()


    for i, image in enumerate(images):
        print(f"tagging {i + 1} of {total_images}")
        # load image and convert to tensor
        tensor = load_image(image, device)
        # tag image and process probs
        with torch.no_grad():
            out = model(tensor)
        probabilities = torch.nn.functional.sigmoid(out[0])

        # convert probs to list, and prune probs below threshold
        prob_list = probabilities.tolist()
        output_tags = {
            tags[index].replace("_", " "): prob
            for index, prob in enumerate(prob_list)
            if prob >= threshold and "placeholder" not in tags[index]
        }

        output_tags = dict(
            sorted(output_tags.items(), key=lambda item: item[1], reverse=True)
        )
        txt_file = Path(image).with_suffix(".txt")

        if txt_file.exists():
            existing_tags = txt_file.read_text().split(', ')
            # Move unknown tags to end of file
            known = []
            unknown = []
            [known.append(x) if x in tags_set else unknown.append(x) for x in existing_tags]
            if unknown:
                unknown = ["*UNKNOWN*"] + unknown

            new_tags = [tag for tag in list(output_tags) if tag not in existing_tags]
            txt_file.write_text(", ".join((prefix + known + ['UNKNOWN'] + unknown + ['NEW_TAGS'] + new_tags)))
        else:
            txt_file.write_text(", ".join((prefix + list(output_tags))))


if __name__ == "__main__":
    main()
