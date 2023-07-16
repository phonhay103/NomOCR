import yaml
import os
import random, string
from tqdm import tqdm
from PIL import Image

CONFIG_PATH = 'data.yaml'
IMAGES_PATH = 'train/images'
LABELS_PATH = 'train/labels'

OUTPUT_IMAGES_PATH = "all_images"
OUTPUT_LABELS_PATH = 'all_images.txt'

with open(CONFIG_PATH, 'r') as file:
    yaml_data = yaml.safe_load(file)
    names = yaml_data.get('names')

os.makedirs(OUTPUT_IMAGES_PATH, exist_ok=True)

label_files = os.listdir(LABELS_PATH)
for label_file in tqdm(label_files):
    image_file = label_file.replace('.txt', '.jpg')

    label_path = os.path.join(LABELS_PATH, label_file)
    image_path = os.path.join(IMAGES_PATH, image_file)

    image = Image.open(image_path)
    image_height, image_width = image.size

    with open(label_path, 'r') as file:
        lines = file.read().splitlines()

    for line in lines:
        class_name = names[int(line.split()[0])]
        _, x, y, h, w = map(float, line.split())
        x_min = int((x - h / 2) * image_height)
        y_min = int((y - w / 2) * image_width)
        x_max = int((x + h / 2) * image_height)
        y_max = int((y + w / 2) * image_width)

        random_string = ''.join(random.choices(string.ascii_letters, k=5))
        cropped_BB_name = f'{random_string}_{class_name}.jpg'
        cropped_BB_path = os.path.join(OUTPUT_IMAGES_PATH, cropped_BB_name)

        cropped_BB = image.crop((x_min, y_min, x_max, y_max))
        cropped_BB.save(cropped_BB_path)

        with open(OUTPUT_LABELS_PATH, 'a') as file:
            file.write(cropped_BB_name + '\t' + class_name + '\n')
