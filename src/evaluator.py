from pathlib import Path

from tqdm import tqdm


def _image_id_from_path(image_path: str) -> int:
    """Extrae el image_id de COCO desde el nombre de archivo (ej. '000000000139.jpg' -> 139)."""
    return int(Path(image_path).stem)


def is_relevant(
    image_path: str,
    target_category_id: int,
    category_index: dict,
) -> bool:
    """Verdadero si la imagen contiene al menos una anotacion con target_category_id."""
    image_id = _image_id_from_path(image_path)
    return target_category_id in category_index.get(image_id, set())


def count_relevant_images(
    target_category_id: int,
    category_index: dict,
) -> int:
    """Cuenta las imagenes del dataset que contienen target_category_id."""
    return sum(1 for cats in category_index.values() if target_category_id in cats)


def precision_at_k(
    results: list[dict],
    target_category_id: int,
    k: int,
    category_index: dict,
) -> float:
    """P@K = |relevantes en top-K| / K"""
    top_k = results[:k]
    if not top_k:
        return 0.0
    relevant = sum(
        1 for r in top_k
        if is_relevant(r["image_path"], target_category_id, category_index)
    )
    return relevant / k


def recall_at_k(
    results: list[dict],
    target_category_id: int,
    k: int,
    category_index: dict,
    total_relevant: int,
) -> float:
    """R@K = |relevantes en top-K| / total_relevantes_en_dataset"""
    if total_relevant == 0:
        return 0.0
    top_k = results[:k]
    relevant = sum(
        1 for r in top_k
        if is_relevant(r["image_path"], target_category_id, category_index)
    )
    return relevant / total_relevant


def evaluate(
    search_engine,
    eval_queries: list[dict],
    category_index: dict,
    k_values: list[int] | None = None,
) -> dict:
    """Ejecuta la evaluacion completa. Retorna dict con resultados por consulta y agregados."""
    if k_values is None:
        k_values = [1, 3, 5]
    max_k = max(k_values)
    query_results = []

    for q in tqdm(eval_queries, desc="Evaluando"):
        results = search_engine.search(q["query_text"], k=max_k)
        target_id = q["target_category_id"]
        total_relevant = count_relevant_images(target_id, category_index)

        metrics = {}
        for k in k_values:
            metrics[f"precision_at_{k}"] = precision_at_k(
                results, target_id, k, category_index
            )
            metrics[f"recall_at_{k}"] = recall_at_k(
                results, target_id, k, category_index, total_relevant
            )

        relevance_flags = [
            is_relevant(r["image_path"], target_id, category_index) for r in results
        ]

        query_results.append(
            {
                "query_text": q["query_text"],
                "target_category": q["target_category"],
                "target_category_id": target_id,
                "total_relevant_in_dataset": total_relevant,
                "results": results,
                "relevance_flags": relevance_flags,
                **metrics,
            }
        )

    aggregates = {}
    for k in k_values:
        aggregates[f"avg_precision_at_{k}"] = (
            sum(r[f"precision_at_{k}"] for r in query_results) / len(query_results)
        )
        aggregates[f"avg_recall_at_{k}"] = (
            sum(r[f"recall_at_{k}"] for r in query_results) / len(query_results)
        )

    success = aggregates.get("avg_precision_at_5", 0.0) >= 0.70

    return {
        "query_results": query_results,
        **aggregates,
        "success": success,
        "success_threshold": 0.70,
        "total_queries": len(query_results),
    }
