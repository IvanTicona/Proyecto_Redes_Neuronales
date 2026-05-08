import numpy as np
import faiss
from pathlib import Path

import src.config as config


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Construye un FAISS IndexFlatIP a partir de embeddings L2-normalizados."""
    index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
    index.add(embeddings)
    return index


def save_index(index: faiss.IndexFlatIP, path) -> None:
    faiss.write_index(index, str(path))


def load_index(path) -> faiss.IndexFlatIP:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Indice FAISS no encontrado en '{path}'. Ejecute scripts/build_index.py primero."
        )
    return faiss.read_index(str(path))


