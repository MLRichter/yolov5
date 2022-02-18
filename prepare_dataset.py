import shutil
from pathlib import Path
from typing import Dict, Union, List

import numpy as np
from skimage.io import imread, imsave
from tqdm import tqdm

import click

CLASS_MAPPING = ["Diabetic Food Syndrome", "Ulcus Cruris Venosum"]


def _setup_target_folder(src: Path, tgt: Path) -> Dict[str, Path]:
    if not src.exists():
        raise ValueError("Source Folder does not exist")
    datasets = [tgt / d.name for d in src.iterdir() if d.is_dir()]
    print("Found the following subsets:", datasets)
    if tgt.exists():
        print("Target folder already exists, skipping")
        overwrite = input("Remove (y/n): ")
    else:
        overwrite = "n"
    if overwrite == "y":
        shutil.rmtree(tgt)
        print("Removed target folder")
    for dataset in datasets:
        print("Creating dataset folder for", str(dataset))
        dataset.mkdir(parents=True, exist_ok=True)
    return {d.name: d for d in datasets}


def find_class(label: Path) -> str:
    with open(label, "r") as f:
        for line in f.readlines():
            #print(label, line)
            return CLASS_MAPPING[int(line.split(" ")[0])]
    raise ValueError("Could not find class for", str(label))


def decode_line(line: str) -> Dict[str, Union[str, float]]:
    line = line.split(" ")
    result = {
        "class": CLASS_MAPPING[int(line[0])],
        "x": float(line[1]),
        "y": float(line[2]),
        "w": float(line[3]),
        "h": float(line[4])
    }
    return result


def denormalize(image: np.ndarray, boxes:  List[Dict[str, Union[str, float]]]) -> List[Dict[str, Union[str, int]]]:
    denormed_boxes = []
    for box in boxes:
        dw, dh = image.shape[1], image.shape[0]
        x, y, w, h = box["x"]*dw, box["y"]*dh, (box["w"]*dw)/2, (box["h"]*dh)/2
        xmin = x - w
        xmax = x + w
        ymin = y - h
        ymax = y + h
        d_box = {
            "xmin": int(xmin),
            "ymin": int(ymin),
            "xmax": int(xmax),
            "ymax": int(ymax),
            "class": box["class"]
        }
        denormed_boxes.append(d_box)
    return denormed_boxes


def find_box_info(src: Path, label: Path) -> List[Dict[str, Union[str, float]]]:
    boxes: List[Dict[str, Union[str, float]]] = []
    with open(label, "r") as f:
        for line in f:
            box = decode_line(line)
            boxes.append(box)
    try:
        img: np.ndarray = imread(str(src))
    except:
        return []
    denormed_boxes = denormalize(img, boxes)
    return denormed_boxes


def crop_bounding_boxes(src: Path, bounding_boxes: List[Dict[str, Union[str, float]]]) -> List[np.ndarray]:
    if not bounding_boxes:
        return []
    img: np.ndarray = imread(str(src))
    cropped_images = []
    for box in bounding_boxes:
        xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
        cropped_image = img[ymin:ymax, xmin:xmax]
        cropped_images.append(cropped_image)
    return cropped_images


def save_images(src: Path, images: List[np.ndarray], tgt: Path, box_infos: List[Dict[str, Union[str, float]]]) -> None:
    original_file_name = src.with_suffiyx("").name
    for i, image in enumerate(images):
        cls = box_infos[i]["class"]
        image_name = original_file_name + cls + "_" + str(i) + ".png"
        target_path = tgt / cls if tgt.name != cls else tgt
        target_path.mkdir(parents=True, exist_ok=True)
        final_target_path = target_path / image_name
        imsave(str(final_target_path), image)


def _process_datapoint_with_crop(src: Path, label: Path, tgt: Path) -> None:
    bounding_boxes = find_box_info(src, label)
    cropped_images = crop_bounding_boxes(src, bounding_boxes)
    save_images(src, cropped_images, tgt, bounding_boxes)


def _process_datapoint_without_crop(src: Path, label: Path, tgt: Path) -> None:
    try:
        cls = find_class(label)
    except ValueError:
        return
    target_path = tgt / cls if tgt.name != cls else tgt
    target_path.mkdir(parents=True, exist_ok=True)
    img_location = target_path / src.name
    shutil.copy(src, img_location)


def _process_datapoint(src: Path, label: Path, tgt: Path, crop: bool) -> Path:
    if crop:
        _process_datapoint_with_crop(src, label, tgt)
    else:
        _process_datapoint_without_crop(src, label, tgt)


def _process_dataset(src: Path, tgt: Path, crop: bool) -> Path:
    image_folder = src / "images"
    label_folder = src / "labels"
    all_images = list(image_folder.glob("**/*"))
    for i, image in enumerate(tqdm(all_images)):
        label = (label_folder / image.name).with_suffix(".txt")
        if not label.exists():
            print("Label does not exist for ", str(image))
            continue
        _process_datapoint(image, label, tgt, crop)


@click.command()
@click.option('--src', type=str)
@click.option('--crop', type=bool)
@click.option('--tgt', type=str)
def main(src: str, crop: bool, tgt: str) -> None:
    src, tgt = Path(src), Path(tgt)
    datasets = _setup_target_folder(src, tgt)
    for name, folder in datasets.items():
        print("Processing", name)
        _process_dataset(src / name, folder, crop)


if __name__ == '__main__':
    main()