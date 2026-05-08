import pickle
from pathlib import Path

import src.config as config
from src.embedder import load_model, encode_text
from src.indexer import load_index


class SearchEngine:
    def __init__(self, index_path=None, image_paths_path=None):
        index_path = Path(index_path or config.FAISS_INDEX_PATH)
        image_paths_path = Path(image_paths_path or config.IMAGE_PATHS_PATH)

        self.index = load_index(index_path)

        if not image_paths_path.exists():
            raise FileNotFoundError(
                f"Rutas de imagenes no encontradas en '{image_paths_path}'. "
                "Ejecute scripts/build_embeddings.py primero."
            )
        with open(image_paths_path, "rb") as f:
            self.image_paths: list[str] = pickle.load(f)

        model, _, tokenizer = load_model()
        self.model = model
        self.tokenizer = tokenizer

    def search(self, query: str, k: int = config.TOP_K_DEFAULT) -> list[dict]:
        """Busca imagenes que coincidan con la consulta. Retorna lista de dicts ordenada por relevancia."""
        if not query or not query.strip():
            raise ValueError("La consulta no puede estar vacia.")
        query_emb = encode_text(query.strip(), self.model, self.tokenizer)
        actual_k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_emb, actual_k)
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
            if idx == -1:
                continue
            results.append(
                {
                    "rank": rank,
                    "image_path": self.image_paths[idx],
                    "score": float(score),
                }
            )
        return results
