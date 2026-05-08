# Buscador Semántico de Imágenes por Texto

Sistema de recuperación de imágenes basado en embeddings multimodales. Dado un texto en lenguaje natural ("dog running on the beach"), recupera las imágenes más relevantes del dataset MS-COCO 2017 Val (5 000 imágenes).

Construido con OpenCLIP ViT-B-32, FAISS y Gradio.

## Arquitectura

1. El usuario ingresa una consulta en texto natural
2. OpenCLIP ViT-B-32 codifica el texto como un embedding L2-normalizado de 512 dimensiones
3. FAISS IndexFlatIP calcula la similitud coseno contra los 5 000 embeddings de imágenes indexados
4. Se retornan los Top-K resultados ordenados por relevancia

| Componente | Decisión | Justificación |
|---|---|---|
| Modelo multimodal | OpenCLIP ViT-B-32 `laion2b_s34b_b79k` | CLIP open-source, zero-shot robusto |
| Índice vectorial | FAISS IndexFlatIP | Búsqueda exacta, determinista, eficiente para 5K vectores |
| Dataset | MS-COCO 2017 Val | 5 000 imágenes, 80 categorías, ground truth oficial |
| Evaluación | Precision@K / Recall@K | Métricas estándar de recuperación de información |

## Estructura del proyecto

```
Proyecto_Redes_Neuronales/
├── src/
│   ├── config.py           # Rutas y constantes del modelo
│   ├── dataset.py          # Parseo de anotaciones COCO
│   ├── embedder.py         # Codificación con OpenCLIP (imágenes y texto)
│   ├── indexer.py          # Construcción y carga del índice FAISS
│   ├── search_engine.py    # Clase SearchEngine (orquestación del pipeline)
│   └── evaluator.py        # Métricas Precision@K / Recall@K
├── scripts/
│   ├── prepare_dataset.py  # Descarga COCO val2017 y genera metadata.csv
│   ├── build_embeddings.py # Codifica todas las imágenes : image_embeddings.npy
│   ├── build_index.py      # Construye el índice FAISS : faiss_index.bin
│   └── run_evaluation.py   # Evalúa 25 consultas e imprime el reporte P@K / R@K
├── notebooks/
│   └── 01_pipeline_exploration.ipynb  # Recorrido académico completo del pipeline
├── docs/
│   └── Informe_Buscador_Semantico.docx
├── app.py                  # Interfaz Gradio
└── requirements.txt
```

## Ejecución

### En Google Colab

El notebook `notebooks/01_pipeline_exploration.ipynb` es completamente autónomo.

1. Abrir el archivo en Google Colab
2. Ejecutar todas las celdas en orden (Entorno de ejecución : Ejecutar todo)

La primera celda instala todas las dependencias. El notebook descarga el dataset, genera los embeddings, construye el índice, ejecuta demos de búsqueda y la evaluación completa.

### En local (sistema completo)

**1. Instalar dependencias**

```bash
pip install -r requirements.txt
```

> Solo CPU: `faiss-cpu` está incluido. Para GPU: reemplazar por `faiss-gpu`.

**2. Descargar y preparar el dataset**

```bash
python scripts/prepare_dataset.py
```

Descarga el split val2017 de MS-COCO (~1 GB imágenes + ~241 MB anotaciones) y genera `data/metadata.csv`.

**3. Generar embeddings de imágenes**

```bash
python scripts/build_embeddings.py
```

Codifica las 5 000 imágenes con OpenCLIP ViT-B-32 en lotes de 32. Genera:
- `embeddings/image_embeddings.npy` - array float32 (5000 x 512)
- `embeddings/image_paths.pkl` - lista de rutas de imágenes

La primera ejecución descarga los pesos de OpenCLIP (~600 MB).

**4. Construir el índice FAISS**

```bash
python scripts/build_index.py
```

Genera `embeddings/faiss_index.bin` (IndexFlatIP, ~10 MB).

**5. Ejecutar evaluación**

```bash
python scripts/run_evaluation.py
```

Evalúa 25 consultas predefinidas sobre 20 categorías de COCO. Imprime P@1, P@3, P@5 y R@K por consulta y promedios agregados. Criterio de éxito: **Precision@5 >= 0.70**.

**6. Lanzar la interfaz Gradio**

```bash
python app.py
```

Abre la interfaz en `http://localhost:7860`. Ingresar una consulta en texto natural, ajustar Top-K y buscar.

## Resultados de evaluación

Evaluado sobre 25 consultas cubriendo 20 categorías de MS-COCO.

| Métrica | Valor |
|---|---|
| Precision@1 | 1.0000 |
| Precision@3 | 0.9600 |
| Precision@5 | **0.9520** OK |
| Recall@1 | 0.0069 |
| Recall@3 | 0.0200 |
| Recall@5 | 0.0331 |

Criterio de éxito: **Precision@5 >= 0.70** - alcanzado (0.9520).

## Requisitos

- Python 3.10+
- ~2 GB en disco para los datos de COCO (excluidos del repo, generados por `prepare_dataset.py`)
- ~600 MB para los pesos de OpenCLIP (cacheados por `open_clip_torch`)
- ~20 MB para embeddings e índice FAISS (excluidos del repo, generados por `build_embeddings.py` y `build_index.py`)
- GPU opcional - la inferencia en CPU funciona pero es más lenta (~10 min para codificar todo)
