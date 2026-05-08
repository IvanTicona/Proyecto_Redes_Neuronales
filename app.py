from pathlib import Path

import gradio as gr

import src.config as config

_engine = None
_startup_error: str | None = None


def _get_engine():
    global _engine, _startup_error
    if _engine is not None:
        return _engine
    try:
        from src.search_engine import SearchEngine
        _engine = SearchEngine()
    except FileNotFoundError as e:
        _startup_error = str(e)
    except Exception as e:
        _startup_error = f"Error inesperado al cargar el modelo: {e}"
    return _engine


def search(query: str, top_k: int):
    engine = _get_engine()

    if engine is None:
        return [], f"Error de inicio: {_startup_error}"

    if not query or not query.strip():
        return [], "Por favor, ingrese una consulta."

    try:
        results = engine.search(query.strip(), k=int(top_k))
    except ValueError as e:
        return [], str(e)
    except Exception as e:
        return [], f"Error en la busqueda: {e}"

    images = []
    for r in results:
        p = Path(r["image_path"])
        if p.exists():
            images.append((str(p), f"#{r['rank']}  puntaje: {r['score']:.3f}"))

    status = f"{len(images)} resultados para: \"{query.strip()}\""
    return images, status


with gr.Blocks(title="Buscador Semantico de Imagenes") as demo:
    gr.Markdown(
        "# Buscador Semantico de Imagenes\n"
        "Busqueda semantica texto a imagen con OpenCLIP ViT-B-32 + FAISS"
    )

    with gr.Row():
        with gr.Column(scale=4):
            query_input = gr.Textbox(
                label="Consulta",
                placeholder='ej. "perro corriendo en la playa"',
                lines=1,
            )
        with gr.Column(scale=1):
            top_k_slider = gr.Slider(
                minimum=1, maximum=20, value=config.TOP_K_DEFAULT, step=1, label="Top-K"
            )

    search_btn = gr.Button("Buscar", variant="primary")
    status_text = gr.Textbox(label="Estado", interactive=False, lines=1)
    gallery = gr.Gallery(label="Resultados", columns=5, height="auto", object_fit="contain")

    search_btn.click(
        fn=search,
        inputs=[query_input, top_k_slider],
        outputs=[gallery, status_text],
    )
    query_input.submit(
        fn=search,
        inputs=[query_input, top_k_slider],
        outputs=[gallery, status_text],
    )


if __name__ == "__main__":
    demo.launch()
