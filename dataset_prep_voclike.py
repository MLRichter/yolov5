import shutil
import xml.etree.ElementTree as ET
from typing import Tuple, List

import yaml
import click
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils.general import download, Path


CLASSES = {"dfu": 0, "ucv": 1}

def convert_label(path: Path, lb_path: Path, image_id: str) -> None:

    def convert_box(size: int, box: List[int]) -> Tuple[int, int, int, int]:
        dw, dh = 1. / size[0], 1. / size[1]
        x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
        return x * dw, y * dh, w * dw, h * dh

    in_file = open(path / f'{image_id}.xml')
    out_file = open(lb_path, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text

        xmlbox = obj.find('bndbox')
        bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
        cls_id = CLASSES[cls]  # class id
        out_file.write(" ".join([str(a) for a in (cls_id, *bb)]) + '\n')


@click.command()
@click.option('--path', type=str)
@click.option('--target', type=str)
@click.option('--split', type=float, default=0.9)
def main(path: str, target: str, split: float) -> None:
    # Download
    dir = Path(path)  # dataset root dir
    target = Path(target)

    image_ids = [xml_path.with_suffix("").name for xml_path in dir.glob("**/*") if xml_path.name.endswith('.xml')]
    train_ids, val_ids = train_test_split(image_ids, train_size=split)
    print(len(train_ids), len(val_ids), len(image_ids))

    for img_ids, mode in [(train_ids, "train"), (val_ids, "val")]:
        imgs_path = target / mode / 'images'
        lbs_path = target / mode / 'labels'
        imgs_path.mkdir(exist_ok=True, parents=True)
        lbs_path.mkdir(exist_ok=True, parents=True)

        for id in tqdm(img_ids, desc='transforming images'):
            f = find_extension(dir/f"{id}.jpg")  # old img path
            lb_path = (lbs_path / id).with_suffix('.txt')  # new label path
            f_path = imgs_path / f.name
            #f.rename(f_path)  # move image
            shutil.copy(f, f_path)
            convert_label(dir, lb_path, id)  # convert labels to YOLO format

def find_extension(path: Path, suffix_list = [".JPG", ".PNG", ".jpg", ".png"]) -> str:
    for suffix in suffix_list:
        if path.with_suffix(suffix).exists():
            return path.with_suffix(suffix)
    raise ValueError(f"{path} does not have a valid extension")



if __name__ == '__main__':
    main()
