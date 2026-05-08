import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import src.config as config
from src.dataset import parse_coco_annotations, build_image_category_index
from src.evaluator import evaluate
from src.search_engine import SearchEngine

# ---------------------------------------------------------------------------
# 25 consultas de evaluacion — cubre 20 categorias distintas de COCO
# ---------------------------------------------------------------------------
EVAL_QUERIES = [
    {"query_text": "dog running on the beach",             "target_category": "dog",           "target_category_id": 18},
    {"query_text": "cat sleeping on a couch",              "target_category": "cat",            "target_category_id": 17},
    {"query_text": "person riding a bicycle",              "target_category": "bicycle",        "target_category_id": 2},
    {"query_text": "car driving on a city street",         "target_category": "car",            "target_category_id": 3},
    {"query_text": "airplane flying in the sky",           "target_category": "airplane",       "target_category_id": 5},
    {"query_text": "elephant in the wild",                 "target_category": "elephant",       "target_category_id": 22},
    {"query_text": "zebra standing on dry grass",          "target_category": "zebra",          "target_category_id": 24},
    {"query_text": "giraffe eating leaves from a tree",    "target_category": "giraffe",        "target_category_id": 25},
    {"query_text": "pizza on a wooden table",              "target_category": "pizza",          "target_category_id": 59},
    {"query_text": "people playing sports outdoors",       "target_category": "person",         "target_category_id": 1},
    {"query_text": "horse galloping in a green field",     "target_category": "horse",          "target_category_id": 19},
    {"query_text": "bird perched on a branch",             "target_category": "bird",           "target_category_id": 16},
    {"query_text": "bus stopped at a city street",         "target_category": "bus",            "target_category_id": 6},
    {"query_text": "boat sailing on the water",            "target_category": "boat",           "target_category_id": 9},
    {"query_text": "person working on a laptop",           "target_category": "laptop",         "target_category_id": 73},
    {"query_text": "food served on a dining table",        "target_category": "dining table",   "target_category_id": 67},
    {"query_text": "glass bottle on a shelf",              "target_category": "bottle",         "target_category_id": 44},
    {"query_text": "chair next to a table indoors",        "target_category": "chair",          "target_category_id": 62},
    {"query_text": "truck driving on a highway",           "target_category": "truck",          "target_category_id": 8},
    {"query_text": "person surfing on ocean waves",        "target_category": "surfboard",      "target_category_id": 42},
    {"query_text": "cow grazing in a green field",         "target_category": "cow",            "target_category_id": 21},
    {"query_text": "baseball player holding a bat",        "target_category": "baseball bat",   "target_category_id": 39},
    {"query_text": "person holding an umbrella in rain",   "target_category": "umbrella",       "target_category_id": 28},
    {"query_text": "skier going down a snowy mountain",    "target_category": "skis",           "target_category_id": 35},
    {"query_text": "sheep on a hillside meadow",           "target_category": "sheep",          "target_category_id": 20},
]

K_VALUES = [1, 3, 5]

# ---------------------------------------------------------------------------

def _print_report(report: dict) -> None:
    print("\n" + "=" * 65)
    print("  REPORTE DE EVALUACION — Busqueda Semantica de Imagenes")
    print("=" * 65)
    print(f"  Consultas evaluadas: {report['total_queries']}")
    print(f"  Valores de K       : {K_VALUES}")
    print()

    # Tabla por consulta
    header = f"  {'Consulta':<45} {'P@1':>5} {'P@3':>5} {'P@5':>5}"
    print(header)
    print("  " + "-" * 63)
    for r in report["query_results"]:
        p1 = r["precision_at_1"]
        p3 = r["precision_at_3"]
        p5 = r["precision_at_5"]
        q = r["query_text"][:43] + ".." if len(r["query_text"]) > 43 else r["query_text"]
        print(f"  {q:<45} {p1:>5.2f} {p3:>5.2f} {p5:>5.2f}")

    print("  " + "-" * 63)
    print(
        f"  {'PROMEDIO':<45} "
        f"{report['avg_precision_at_1']:>5.2f} "
        f"{report['avg_precision_at_3']:>5.2f} "
        f"{report['avg_precision_at_5']:>5.2f}"
    )
    print()
    print(f"  Recall@1: {report['avg_recall_at_1']:.4f}")
    print(f"  Recall@3: {report['avg_recall_at_3']:.4f}")
    print(f"  Recall@5: {report['avg_recall_at_5']:.4f}")
    print()

    threshold = report["success_threshold"]
    avg_p5 = report["avg_precision_at_5"]
    verdict = "EXITO" if report["success"] else "DEBAJO DEL UMBRAL"
    print(f"  Precision@5 = {avg_p5:.4f}  (umbral: {threshold})  -> {verdict}")
    print()

    # Consultas debajo del umbral
    failures = [
        r for r in report["query_results"] if r["precision_at_5"] < threshold
    ]
    if failures:
        print(f"  Consultas debajo del umbral P@5 ({len(failures)}):")
        for r in failures:
            print(f"    · \"{r['query_text']}\"  ->  P@5={r['precision_at_5']:.2f}")
    print("=" * 65 + "\n")


def main():
    if not config.ANNOTATIONS_PATH.exists():
        raise FileNotFoundError(
            f"Anotaciones no encontradas en '{config.ANNOTATIONS_PATH}'. "
            "Ejecute scripts/prepare_dataset.py primero."
        )
    if not config.FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Indice FAISS no encontrado en '{config.FAISS_INDEX_PATH}'. "
            "Ejecute scripts/build_index.py primero."
        )

    print("Cargando anotaciones COCO ...")
    annotations = parse_coco_annotations(config.ANNOTATIONS_PATH)
    category_index = build_image_category_index(annotations)

    print(
        f"Cargando OpenCLIP {config.MODEL_NAME} ({config.PRETRAINED}) "
        f"en {config.DEVICE} ..."
    )
    engine = SearchEngine()

    print(f"Ejecutando evaluacion sobre {len(EVAL_QUERIES)} consultas ...")
    report = evaluate(engine, EVAL_QUERIES, category_index, k_values=K_VALUES)

    _print_report(report)

    out_path = "evaluation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        # la lista de resultados tiene image_paths, mantener serializable
        json.dump(report, f, indent=2, default=str)
    print(f"Reporte completo guardado -> {out_path}")


if __name__ == "__main__":
    main()
