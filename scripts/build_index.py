import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import src.config as config
from src.indexer import build_index, save_index


def main():
    if not config.IMAGE_EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Embeddings no encontrados en '{config.IMAGE_EMBEDDINGS_PATH}'. "
            "Ejecute scripts/build_embeddings.py primero."
        )

    print(f"Cargando embeddings desde {config.IMAGE_EMBEDDINGS_PATH} ...")
    embeddings = np.load(config.IMAGE_EMBEDDINGS_PATH)
    print(f"Embeddings cargados: {embeddings.shape[0]} vectores de dimension {embeddings.shape[1]}.")

    print("Construyendo indice FAISS IndexFlatIP ...")
    index = build_index(embeddings)
    print(f"El indice contiene {index.ntotal} vectores.")

    print(f"Guardando indice -> {config.FAISS_INDEX_PATH}")
    save_index(index, config.FAISS_INDEX_PATH)
    print("Completado.")


if __name__ == "__main__":
    main()
