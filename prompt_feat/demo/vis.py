import glob
import os
from PIL import Image, ImageDraw


# draw boxes
def draw_boxes(img_path, boxes, labels=None, color=(250,0,30)):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=10)
        if labels:
            draw.rectangle(((x1, y1), (x1 + 50, y1 + 10)), fill=color)
            info = labels[i]
            draw.text((x1, y1), info)
    return img

# draw colored mask
# img.show() can show image
# img.save(path) can save to given path
def draw_rectangles(self, img_path, boxes, color=(240,0,30,127)):
    img = Image.open(img_path).convert("RGB")
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        foreground = Image.new('RGBA', (x2 - x1, y2 - y1), color=color)
        img.paste(foreground, (x1, y1), foreground)
    return img