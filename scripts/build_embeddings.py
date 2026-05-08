import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle

import numpy as np
import pandas as pd

import src.config as config
from src.embedder import load_model, encode_images


def main():
    if not config.METADATA_PATH.exists():
        raise FileNotFoundError(
            f"metadata.csv no encontrado en '{config.METADATA_PATH}'. "
            "Ejecute scripts/prepare_dataset.py primero."
        )

    config.EMBEDDINGS_DIR.mkdir(exist_ok=True)

    print("Cargando metadata ...")
    df = pd.read_csv(config.METADATA_PATH)
    image_paths = df["path"].tolist()
    print(f"Se encontraron {len(image_paths)} imagenes.")

    print(
        f"Cargando OpenCLIP {config.MODEL_NAME} ({config.PRETRAINED}) "
        f"en {config.DEVICE} ..."
    )
    model, preprocess, _ = load_model()

    print("Codificando imagenes (puede tardar varios minutos en CPU) ...")
    embeddings = encode_images(image_paths, model, preprocess)

    print(f"Guardando embeddings {embeddings.shape} -> {config.IMAGE_EMBEDDINGS_PATH}")
    np.save(config.IMAGE_EMBEDDINGS_PATH, embeddings)

    with open(config.IMAGE_PATHS_PATH, "wb") as f:
        pickle.dump(image_paths, f)
    print(f"Rutas de imagenes guardadas -> {config.IMAGE_PATHS_PATH}")
    print("Completado.")


if __name__ == "__main__":
    main()
