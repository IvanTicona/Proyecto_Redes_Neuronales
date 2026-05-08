import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import shutil
import zipfile

import pandas as pd
import requests
from tqdm import tqdm

import src.config as config

COCO_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)


def download_file(url: str, dest) -> None:
    print(f"Descargando {url} ...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def main():
    config.DATA_DIR.mkdir(exist_ok=True)

    # --- Imagenes ---
    images_zip = config.DATA_DIR / "val2017.zip"
    extracted_images = config.DATA_DIR / "val2017"

    if not config.IMAGES_DIR.exists() or not any(config.IMAGES_DIR.iterdir()):
        if not images_zip.exists():
            download_file(COCO_IMAGES_URL, images_zip)
        print("Extrayendo imagenes ...")
        with zipfile.ZipFile(images_zip, "r") as zf:
            zf.extractall(config.DATA_DIR)
        # El zip de COCO extrae a val2017/, se renombra a images/
        if extracted_images.exists():
            if config.IMAGES_DIR.exists():
                config.IMAGES_DIR.rmdir()
            shutil.move(str(extracted_images), str(config.IMAGES_DIR))
        print(f"Imagenes listas en {config.IMAGES_DIR}")
    else:
        print(f"Imagenes ya presentes en {config.IMAGES_DIR}, omitiendo descarga.")

    # --- Anotaciones ---
    annotations_zip = config.DATA_DIR / "annotations_trainval2017.zip"

    if not config.ANNOTATIONS_PATH.exists():
        if not annotations_zip.exists():
            download_file(COCO_ANNOTATIONS_URL, annotations_zip)
        print("Extrayendo anotaciones ...")
        with zipfile.ZipFile(annotations_zip, "r") as zf:
            zf.extractall(config.DATA_DIR)
        print(f"Anotaciones listas en {config.ANNOTATIONS_PATH}")
    else:
        print("Anotaciones ya presentes, omitiendo descarga.")

    # --- metadata.csv ---
    print("Generando metadata.csv ...")
    with open(config.ANNOTATIONS_PATH, encoding="utf-8") as f:
        coco = json.load(f)

    cat_map = {cat["id"]: cat["name"] for cat in coco["categories"]}

    image_cats: dict[int, set[str]] = {}
    image_cat_ids: dict[int, set[int]] = {}
    for ann in coco["annotations"]:
        iid = ann["image_id"]
        cid = ann["category_id"]
        image_cats.setdefault(iid, set()).add(cat_map.get(cid, "unknown"))
        image_cat_ids.setdefault(iid, set()).add(cid)

    rows = []
    for img in coco["images"]:
        iid = img["id"]
        path = str(config.IMAGES_DIR / img["file_name"])
        cats = sorted(image_cats.get(iid, set()))
        cat_ids = sorted(image_cat_ids.get(iid, set()))
        rows.append(
            {
                "image_id": iid,
                "path": path,
                "categories": "|".join(cats),
                "category_ids": "|".join(str(c) for c in cat_ids),
                "optional_description": "",
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(config.METADATA_PATH, index=False)
    print(f"metadata.csv guardado: {len(df)} imagenes en {len(cat_map)} categorias.")


if __name__ == "__main__":
    main()
