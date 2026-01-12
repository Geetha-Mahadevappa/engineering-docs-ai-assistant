"""
finetune_embeddings.py

Two‑stage fine‑tuning pipeline for multilingual E5 embeddings:

Stage 1:
    - Preprocess into (query, positive)
    - Fine‑tune with MNLR
    - Build FAISS index

Stage 2:
    - Mine hard negatives using Stage‑1 model
    - Fine‑tune again with (query, positive, hard_negative)
    - Build final FAISS index

All configuration values are read from configs/embedded_training.yaml.
"""

import os
import yaml
import pandas as pd
import faiss

from training import download_data
from training import preprocessing
from training import trainer
from training.hard_negative_mining import HardNegativeMiner
from inference.model import EmbeddingModel


class EmbeddingFineTunePipeline:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.kaggle_path = self.config["kaggle_data"]
        self.train_pairs_stage1 = self.config["train_pairs"]
        self.train_pairs_stage2 = self.config["train_pairs_hard"]

        # Stage‑1 and Stage‑2 model directories
        base = self.config["model_output_path"]
        self.model_stage1 = base + "_stage1"
        self.model_stage2 = base + "_stage2"

        # FAISS index paths
        self.index_stage1 = os.path.join(self.model_stage1, "faiss.index")
        self.index_stage2 = os.path.join(self.model_stage2, "faiss.index")

    # Load documents for FAISS indexing
    def load_documents_for_indexing(self, limit: int = 5000):
        questions_file = os.path.join(self.kaggle_path, "Questions.csv")
        df = pd.read_csv(questions_file, encoding="latin1", engine="python", on_bad_lines="skip")
        docs = (df["Title"].fillna("") + "\n" + df["Body"].fillna("")).tolist()
        return docs[:limit]

    # Build FAISS HNSW index
    def build_faiss_index(self, model_path: str, documents: list, index_path: str):
        print(f"Loading model from {model_path}")
        model = EmbeddingModel(model_path)

        print("Encoding documents...")
        vectors = model.encode(documents, batch_size=self.config["batch_size"]).numpy().astype("float32")

        dim = vectors.shape[1]
        print(f"Building FAISS HNSW index (dim={dim})")

        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 128
        index.add(vectors)

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(index, index_path)

        print(f"FAISS index saved to {index_path}")

    # Run full two‑stage pipeline
    def run(self):
        print("Download dataset")
        download_data.download_stacksample()

        print("Preprocess dataset")
        preprocessing.preprocess_dataset(self.kaggle_path, self.train_pairs_stage1)

        print("Fine‑tune model (Stage‑1)")
        self.config["train_pairs"] = self.train_pairs_stage1
        self.config["model_output_path"] = self.model_stage1
        trainer.EmbeddingTrainer(self.config).train()

        print("Build FAISS index (Stage‑1)")
        docs = self.load_documents_for_indexing()
        self.build_faiss_index(self.model_stage1, docs, self.index_stage1)

        print("Mine hard negatives (Stage‑2)")
        miner = HardNegativeMiner(
            model_path=self.model_stage1,
            kaggle_path=self.kaggle_path,
            output_jsonl=self.train_pairs_stage2
        )
        miner.run()

        print("Fine‑tune with hard negatives (Stage‑2)")
        self.config["train_pairs"] = self.train_pairs_stage2
        self.config["model_output_path"] = self.model_stage2
        trainer.EmbeddingTrainer(self.config).train()

        print("Build final FAISS index (Stage‑2)")
        self.build_faiss_index(self.model_stage2, docs, self.index_stage2)

        print("Pipeline complete. Stage‑2 model and FAISS index are ready.")


if __name__ == "__main__":
    pipeline = EmbeddingFineTunePipeline("configs/embedded_training.yaml")
    pipeline.run()
