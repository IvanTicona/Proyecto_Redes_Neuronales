import json


def parse_coco_annotations(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_image_category_index(annotations: dict) -> dict[int, set[int]]:
    """Returns {image_id: {category_id, ...}} for fast lookup."""
    index: dict[int, set[int]] = {}
    for ann in annotations["annotations"]:
        index.setdefault(ann["image_id"], set()).add(ann["category_id"])
    return index
