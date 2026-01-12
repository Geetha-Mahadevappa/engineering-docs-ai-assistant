"""
trainer.py

Fine‑tune a multilingual E5 embedding model using Multiple Negatives Ranking Loss (MNLR).
This script supports both single‑GPU and multi‑GPU training depending on how it is launched.
Configuration is loaded from config.yaml.

Usage:
    python trainer.py
    torchrun --nproc_per_node=4 trainer.py
"""

import os
import json
import yaml
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModel, AutoTokenizer
from torch.cuda.amp import autocast, GradScaler


# Dataset
class QueryPositiveDataset(Dataset):
    """Simple dataset for (query, positive) pairs."""

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 256):
        self.items = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Training file not found: {jsonl_path}")

        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    if "query" in item and "positive" in item:
                        self.items.append(item)
        except Exception as e:
            raise RuntimeError(f"Failed to load training data: {e}")

        if len(self.items) == 0:
            raise ValueError("Training dataset is empty or invalid.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return {
            "query": item["query"],
            "positive": item["positive"],
        }


# MNLR Loss
def mnlr_loss(q_emb, p_emb, temperature=0.05):
    """
    Multiple Negatives Ranking Loss.
    Each query is matched with its positive, and all other positives in the batch
    act as negatives.
    """
    # Normalize embeddings
    q = torch.nn.functional.normalize(q_emb, p=2, dim=1)
    p = torch.nn.functional.normalize(p_emb, p=2, dim=1)

    # Similarity matrix: (batch_size x batch_size)
    logits = torch.matmul(q, p.T) / temperature

    # Labels: each query matches its own positive (diagonal)
    labels = torch.arange(logits.size(0), device=logits.device)

    return torch.nn.functional.cross_entropy(logits, labels)


# Mean Pooling (E5-style)
def mean_pooling(model_output, attention_mask):
    """Mean pooling over the token embeddings."""
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).float()

    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)

    return summed / counts


# Trainer Class
class EmbeddingTrainer:
    """Clean class-based trainer for MNLR fine-tuning."""

    def __init__(self, config):
        self.config = config
        self.device = self._setup_device()
        self._setup_distributed()

        # AMP / precision settings
        self.use_fp16 = bool(config.get("fp16", False))
        self.use_bf16 = bool(config.get("bf16", False))
        self.grad_accum = int(config.get("grad_accumulation", 1))

        self.use_amp = self.device.type == "cuda" and (self.use_fp16 or self.use_bf16)

        if self.use_bf16 and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(self.local_rank)
            # Use bfloat16 only on Ampere+ (sm >= 80)
            self.amp_dtype = torch.bfloat16 if props.major >= 8 else torch.float16
        else:
            self.amp_dtype = torch.float16

        self.scaler = GradScaler(enabled=self.use_amp)

        # Load model + tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
            self.model = AutoModel.from_pretrained(config["model_name"])
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{config['model_name']}': {e}")

        self.model.to(self.device)

        if self._is_distributed():
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        # Dataset + DataLoader
        dataset = QueryPositiveDataset(
            config["train_pairs"],
            self.tokenizer,
            max_length=config.get("max_length", 256)
        )

        sampler = (
            torch.utils.data.distributed.DistributedSampler(dataset)
            if self._is_distributed()
            else None
        )

        self.dataloader = DataLoader(
            dataset,
            batch_size=config.get("batch_size", 32),
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=2,
        )

        self.sampler = sampler
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.get("learning_rate", 1e-5))

    # Distributed helpers
    def _setup_device(self):
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.local_rank}")
        return torch.device("cpu")

    def _setup_distributed(self):
        if "RANK" in os.environ:
            dist.init_process_group(backend="nccl")

    def _is_distributed(self):
        return dist.is_available() and dist.is_initialized()

    # Training Loop
    def train(self):
        epochs = self.config.get("epochs", 2)
        temperature = self.config.get("temperature", 0.05)
        max_length = self.config.get("max_length", 256)

        self.model.train()
        self.optimizer.zero_grad()

        for epoch in range(1, epochs + 1):
            if self.sampler:
                self.sampler.set_epoch(epoch)

            total_loss = 0.0

            for step, batch in enumerate(self.dataloader):
                # E5 prefixing
                q_texts = [f"query: {q}" for q in batch["query"]]
                p_texts = [f"passage: {p}" for p in batch["positive"]]

                with autocast(dtype=self.amp_dtype, enabled=self.use_amp):
                    # Tokenize
                    q_tok = self.tokenizer(
                        q_texts,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt"
                    ).to(self.device)

                    p_tok = self.tokenizer(
                        p_texts,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt"
                    ).to(self.device)

                    # Forward
                    q_out = self.model(**q_tok)
                    p_out = self.model(**p_tok)

                    q_emb = mean_pooling(q_out, q_tok["attention_mask"])
                    p_emb = mean_pooling(p_out, p_tok["attention_mask"])

                    # DDP-aware MNLR: use all GPUs as negatives
                    if self._is_distributed():
                        # Gather embeddings from all ranks
                        q_list = [torch.zeros_like(q_emb) for _ in range(self.world_size)]
                        p_list = [torch.zeros_like(p_emb) for _ in range(self.world_size)]

                        dist.all_gather(q_list, q_emb)
                        dist.all_gather(p_list, p_emb)

                        q_all = torch.cat(q_list, dim=0)
                        p_all = torch.cat(p_list, dim=0)

                        # Normalize
                        q_all = torch.nn.functional.normalize(q_all, p=2, dim=1)
                        p_all = torch.nn.functional.normalize(p_all, p=2, dim=1)

                        # Similarity matrix over all queries vs all positives
                        logits = torch.matmul(q_all, p_all.T) / temperature

                        local_bs = q_emb.size(0)
                        start = self.rank * local_bs
                        end = start + local_bs

                        # Labels point to the correct positive example in the global batch
                        labels = torch.arange(start, end, device=logits.device)

                        # Only use loss for this rank's queries vs all positives
                        logits_local = logits[start:end]
                        loss = torch.nn.functional.cross_entropy(logits_local, labels)
                    else:
                        # Fallback to single-GPU MNLR loss calculation
                        loss = mnlr_loss(q_emb, p_emb, temperature=temperature)

                    # Scale loss for gradient accumulation
                    loss = loss / self.grad_accum

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                total_loss += loss.item() * self.grad_accum

                # Optimizer step and gradient zeroing on accumulation boundaries
                is_update_step = ((step + 1) % self.grad_accum == 0) or ((step + 1) == len(self.dataloader))
                if is_update_step:
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad()

                if self.rank == 0 and (step + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch}/{epochs} | "
                        f"Step {step + 1}/{len(self.dataloader)} | "
                        f"Loss: {loss.item() * self.grad_accum:.4f}"
                    )

            if self.rank == 0:
                avg_epoch_loss = total_loss / len(self.dataloader)
                print(f"\n--- Epoch {epoch} finished. Avg Loss: {avg_epoch_loss:.4f} ---\n")
                self._save_model()

    def _save_model(self):
        os.makedirs(self.config["model_output_path"], exist_ok=True)
        save_path = self.config["model_output_path"]

        model_to_save = self.model.module if self._is_distributed() else self.model
        model_to_save.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        print(f"Model saved to {save_path}")


if __name__ == "__main__":
    with open("configs/embedded_training.yaml", "r") as f:
        config = yaml.safe_load(f)

    trainer = EmbeddingTrainer(config)
    trainer.train()
