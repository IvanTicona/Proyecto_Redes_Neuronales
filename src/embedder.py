import numpy as np
import torch
import open_clip
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import src.config as config


def load_model():
    """Carga el modelo OpenCLIP ViT-B-32, preprocesador y tokenizador."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.MODEL_NAME, pretrained=config.PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(config.MODEL_NAME)
    model = model.to(config.DEVICE).eval()
    return model, preprocess, tokenizer


class _ImageDataset(Dataset):
    def __init__(self, image_paths: list[str], preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.preprocess(img)


def encode_images(image_paths: list[str], model, preprocess) -> np.ndarray:
    """Codifica imagenes a embeddings L2-normalizados. Retorna array float32 (N, 512)."""
    dataset = _ImageDataset(image_paths, preprocess)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, num_workers=0)
    all_embeddings = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Codificando imagenes"):
            batch = batch.to(config.DEVICE)
            features = model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
            all_embeddings.append(features.cpu().numpy())
    return np.vstack(all_embeddings).astype(np.float32)


def encode_text(query: str, model, tokenizer) -> np.ndarray:
    """Codifica una consulta de texto a embedding L2-normalizado. Retorna array float32 (1, 512)."""
    tokens = tokenizer([query]).to(config.DEVICE)
    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype(np.float32)
