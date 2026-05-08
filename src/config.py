import torch
from pathlib import Path

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
EMBEDDING_DIM = 512
TOP_K_DEFAULT = 5
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"
METADATA_PATH = DATA_DIR / "metadata.csv"
ANNOTATIONS_PATH = DATA_DIR / "annotations" / "instances_val2017.json"

EMBEDDINGS_DIR = Path("embeddings")
IMAGE_EMBEDDINGS_PATH = EMBEDDINGS_DIR / "image_embeddings.npy"
IMAGE_PATHS_PATH = EMBEDDINGS_DIR / "image_paths.pkl"
FAISS_INDEX_PATH = EMBEDDINGS_DIR / "faiss_index.bin"
