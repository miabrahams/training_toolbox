from PIL import Image, ImageDraw, ImageFont
from typing import List
import math

def grid(images: List[Image.Image], labels = None, width = 0, height = 0, border = 0, square = False, horizontal = False, vertical = False): # pylint: disable=redefined-outer-name
    def wrap(text: str, font: ImageFont.FreeTypeFont, length: int):
        lines = ['']
        for word in text.split():
            line = f'{lines[-1]} {word}'.strip()
            if font.getlength(line) <= length:
                lines[-1] = line
            else:
                lines.append(word)
        return '\n'.join(lines)
    if horizontal:
        rows = 1
    elif vertical:
        rows = len(images)
    elif square:
        rows = round(math.sqrt(len(images)))
    else:
        rows = math.floor(math.sqrt(len(images)))
    cols = math.ceil(len(images) / rows)
    size = [0, 0]
    if width == 0:
        w = max([i.size[0] for i in images])
        size[0] = cols * w + cols * border
    else:
        size[0] = width
        w = round(width / cols)
    if height == 0:
        h = max([i.size[1] for i in images])
        size[1] = rows * h + rows * border
    else:
        size[1] = height
        h = round(height / rows)
    # size = tuple(size)
    image = Image.new('RGB', size = size, color = 'black') # pylint: disable=redefined-outer-name
    # Font size
    font = ImageFont.truetype('DejaVuSansMono', round(w / 40))
    for i, img in enumerate(images): # pylint: disable=redefined-outer-name
        x = (i % cols * w) + (i % cols * border)
        y = (i // cols * h) + (i // cols * border)
        img.thumbnail((w, h), Image.Resampling.HAMMING)
        image.paste(img, box=(x, y))
        if labels is not None and len(images) == len(labels):
            ctx = ImageDraw.Draw(image)
            label = wrap(labels[i], font, w)
            # Shadow
            ctx.text((x + 1 + round(w / 400), y + 1 + round(w / 400)), label, font = font, fill = (0, 0, 0))
            # Main text
            ctx.text((x, y), label, font = font, fill = (255, 255, 255))
    return image
