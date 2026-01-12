"""
A small inference‑time wrapper around the fine‑tuned multilingual E5 model.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModel


class EmbeddingModel:
    def __init__(self, model_path: str, device: str = None):
        """
        Load the fine‑tuned model from a local directory.
        The directory must contain the files saved by the trainer.
        """
        if not os.path.isdir(model_path):
            raise ValueError(
                f"Expected a local fine‑tuned model directory, but got: {model_path}"
            )

        # Decide which device to use
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load model in INT8 using bitsandbytes
        self.model = AutoModel.from_pretrained(model_path, load_in_8bit=True, device_map="auto")

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts, batch_size: int = 32):
        """
        Convert a list of texts into embeddings using the fine‑tuned model.
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            raise ValueError("No texts provided for embedding.")

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model(**tokens)
            embeddings = outputs.pooler_output

            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)
