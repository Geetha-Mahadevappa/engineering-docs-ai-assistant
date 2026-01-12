"""
hard_negative_mining.py

Generate hard negatives for Stage‑2 fine‑tuning.

This script:
    1. Loads the Stage‑1 fine‑tuned embedding model
    2. Encodes all StackOverflow answers
    3. Builds a temporary FAISS index
    4. For each (query, positive) pair:
           - embeds the query
           - retrieves top‑k similar answers
           - skips the true positive using pid
           - selects the closest incorrect answer as the hard negative
    5. Writes a new JSONL file:
           { query, positive, hard_negative, qid, pid }

This dataset is used for Stage‑2 fine‑tuning.
"""

import os
import json
import faiss
import pandas as pd
from inference.model import EmbeddingModel


class HardNegativeMiner:
    def __init__(self, model_path: str, kaggle_path: str, output_jsonl: str, top_k: int = 10):
        self.model_path = model_path
        self.kaggle_path = kaggle_path
        self.output_jsonl = output_jsonl
        self.top_k = top_k

    # Load raw StackOverflow data
    def load_raw(self):
        questions_file = os.path.join(self.kaggle_path, "Questions.csv")
        answers_file = os.path.join(self.kaggle_path, "Answers.csv")

        questions = pd.read_csv(
            questions_file,
            encoding="latin1",
            engine="python",
            on_bad_lines="skip"
        )

        answers = pd.read_csv(
            answers_file,
            encoding="latin1",
            engine="python",
            on_bad_lines="skip"
        )

        return questions, answers

    # Build FAISS index over all answers
    def build_answer_index(self, answers):
        print("Encoding all answers...")
        answer_texts = [f"passage: {body}" for body in answers["Body"].fillna("").tolist()]
        answer_ids = answers["Id"].tolist()

        model = EmbeddingModel(self.model_path)
        answer_emb = model.encode(answer_texts, batch_size=32).numpy().astype("float32")

        dim = answer_emb.shape[1]
        print(f"Building temporary FAISS index (dim={dim})")

        index = faiss.IndexFlatL2(dim)
        index.add(answer_emb)

        return index, answer_emb, answer_ids

    # Mine hard negatives
    def run(self):
        print("Loading StackOverflow data...")
        questions, answers = self.load_raw()

        print("Building FAISS index over answers...")
        index, answer_emb, answer_ids = self.build_answer_index(answers)

        # O(1) lookup for answer bodies
        id_to_body = dict(zip(answers["Id"], answers["Body"]))

        print(f"Mining hard negatives (top_k={self.top_k})...")
        os.makedirs(os.path.dirname(self.output_jsonl), exist_ok=True)

        model = EmbeddingModel(self.model_path)

        with open(self.output_jsonl, "w", encoding="utf-8") as out:
            for _, q in questions.iterrows():
                qid = int(q["Id"])

                # Find the positive answer
                pos_rows = answers[answers["ParentId"] == qid]
                if pos_rows.empty:
                    continue

                pos_row = pos_rows.sort_values("Score", ascending=False).iloc[0]
                pos_id = int(pos_row["Id"])
                pos_text = f"passage: {pos_row['Body']}"

                # Build query text
                title = str(q["Title"]) if pd.notna(q["Title"]) else ""
                body = str(q["Body"]) if pd.notna(q["Body"]) else ""
                query_text = f"query: {title}\n{body}"

                # Embed query
                q_emb = model.encode([query_text]).numpy().astype("float32")

                # Retrieve top‑k candidates
                D, I = index.search(q_emb, self.top_k)

                # Select closest incorrect answer
                hard_neg_text = None
                for idx in I[0]:
                    candidate_id = answer_ids[idx]
                    if candidate_id != pos_id:
                        hard_neg_body = id_to_body.get(candidate_id, "")
                        hard_neg_text = f"passage: {hard_neg_body}"
                        break

                if hard_neg_text is None:
                    continue  # no valid negative found

                # Write JSONL
                out.write(json.dumps({
                    "query": query_text,
                    "positive": pos_text,
                    "hard_negative": hard_neg_text,
                    "qid": qid,
                    "pid": pos_id
                }) + "\n")

        print(f"Hard‑negative dataset saved to {self.output_jsonl}")
