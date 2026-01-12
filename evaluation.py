"""
evaluate.py

Evaluate a fine‑tuned embedding model on the StackSample dataset using:
    - Recall@1
    - Recall@5
    - Recall@10
    - MRR (Mean Reciprocal Rank)

This script:
    1. Loads the fine‑tuned model
    2. Loads Questions.csv and Answers.csv
    3. Embeds all answers
    4. Embeds each query
    5. Retrieves top‑K answers using FAISS
    6. Computes evaluation metrics

Run independently after training:
    python evaluate.py
"""

import os
import faiss
import yaml
import pandas as pd
from inference.model import EmbeddingModel


class EmbeddingEvaluator:
    def __init__(self, config_path: str, model_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.kaggle_path = self.config["kaggle_data"]
        self.model_path = model_path
        self.top_k_values = [1, 5, 10]

    # Load StackOverflow data
    def load_raw(self):
        q_path = os.path.join(self.kaggle_path, "Questions.csv")
        a_path = os.path.join(self.kaggle_path, "Answers.csv")

        questions = pd.read_csv(q_path, encoding="latin1", engine="python", on_bad_lines="skip")
        answers = pd.read_csv(a_path, encoding="latin1", engine="python", on_bad_lines="skip")

        return questions, answers

    # Build FAISS index over all answers
    def build_answer_index(self, answers):
        print("Encoding all answers...")
        answer_texts = [f"passage: {body}" for body in answers["Body"].fillna("").tolist()]
        answer_ids = answers["Id"].tolist()

        model = EmbeddingModel(self.model_path)
        answer_emb = model.encode(answer_texts, batch_size=32).numpy().astype("float32")

        dim = answer_emb.shape[1]
        print(f"Building FAISS index (dim={dim})")

        index = faiss.IndexFlatL2(dim)
        index.add(answer_emb)

        return index, answer_emb, answer_ids

    # Compute metrics
    def evaluate(self):
        print("Loading data...")
        questions, answers = self.load_raw()

        print("Building FAISS index...")
        index, answer_emb, answer_ids = self.build_answer_index(answers)

        model = EmbeddingModel(self.model_path)

        recall_counts = {k: 0 for k in self.top_k_values}
        mrr_total = 0
        total = 0

        print("Evaluating model...")
        for _, q in questions.iterrows():
            qid = int(q["Id"])

            # Find positive answer
            pos_rows = answers[answers["ParentId"] == qid]
            if pos_rows.empty:
                continue

            pos_row = pos_rows.sort_values("Score", ascending=False).iloc[0]
            pos_id = int(pos_row["Id"])

            # Build query text
            title = str(q["Title"]) if pd.notna(q["Title"]) else ""
            body = str(q["Body"]) if pd.notna(q["Body"]) else ""
            query_text = f"query: {title}\n{body}"

            # Embed query
            q_emb = model.encode([query_text]).numpy().astype("float32")

            # Retrieve top‑10
            D, I = index.search(q_emb, 10)
            retrieved_ids = [answer_ids[idx] for idx in I[0]]

            # Compute metrics
            total += 1

            # Recall@K
            for k in self.top_k_values:
                if pos_id in retrieved_ids[:k]:
                    recall_counts[k] += 1

            # MRR
            if pos_id in retrieved_ids:
                rank = retrieved_ids.index(pos_id) + 1
                mrr_total += 1 / rank

        # Final metrics
        print("Evaluation Results")
        for k in self.top_k_values:
            recall = recall_counts[k] / total
            print(f"Recall@{k}: {recall:.4f}")

        mrr = mrr_total / total
        print(f"MRR: {mrr:.4f}")
        print(f"Evaluated on {total} queries.")


if __name__ == "__main__":
    # Evaluate the Stage‑2 model
    evaluator = EmbeddingEvaluator(
        config_path="configs/embedded_training.yaml",
        model_path="models/fine_tuned_e5_stage2"
    )
    evaluator.evaluate()
