"""
Preprocess the StackSample dataset to create training pairs for embedding fine‑tuning.

This script loads the Questions.csv and Answers.csv files, matches each question
with its highest‑scoring answer, and generates simple (query, positive) pairs.
The output is saved as data/processed/train_pairs.jsonl.

The goal is to give the embedding model examples of how technical questions
relate to their correct answers.
"""

import os
import json
import pandas as pd


def load_raw_data(data_dir: str):
    """Load Questions.csv and Answers.csv with fallback encoding and skip problematic rows."""
    try:
        questions_path = os.path.join(data_dir, "Questions.csv")
        answers_path = os.path.join(data_dir, "Answers.csv")

        if not os.path.exists(questions_path):
            raise FileNotFoundError(f"Questions.csv not found at: {questions_path}")

        if not os.path.exists(answers_path):
            raise FileNotFoundError(f"Answers.csv not found at: {answers_path}")

        questions = pd.read_csv(
            questions_path,
            encoding="latin1",
            engine="python",
            on_bad_lines="skip"
        )

        answers = pd.read_csv(
            answers_path,
            encoding="latin1",
            engine="python",
            on_bad_lines="skip"
        )

        return questions, answers

    except Exception as e:
        print(f"[ERROR] Failed to load raw data: {e}")
        raise


def build_training_pairs(questions, answers):
    """
    Create (query, positive) pairs using ParentId.

    - query: question title + body
    - positive: highest‑scoring answer for that question
    - qid: question ID (needed for hard‑negative mining)
    - pid: positive answer ID (needed for hard‑negative mining)
    """
    try:
        pairs = []
        grouped = answers.groupby("ParentId")

        for _, q in questions.iterrows():
            qid = int(q["Id"])

            # Skip questions with no answers
            if qid not in grouped.groups:
                continue

            q_answers = grouped.get_group(qid)

            # Pick the highest‑scoring answer as the positive example
            positive_row = q_answers.sort_values("Score", ascending=False).iloc[0]
            positive_text = str(positive_row["Body"])
            pid = int(positive_row["Id"])

            # Combine title + body, preserving HTML/code blocks
            title = str(q["Title"]) if pd.notna(q["Title"]) else ""
            body = str(q["Body"]) if pd.notna(q["Body"]) else ""
            query_text = f"{title}\n{body}"

            # E5 requires prefixing for best performance
            pairs.append({
                "query": f"query: {query_text}",
                "positive": f"passage: {positive_text}",
                "qid": qid,
                "pid": pid
            })

        if not pairs:
            raise ValueError("No training pairs were generated.")

        return pairs

    except Exception as e:
        print(f"[ERROR] Failed to build training pairs: {e}")
        raise


def save_pairs(pairs: list[dict], output_path: str) -> None:
    """Save training pairs as a JSONL file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for item in pairs:
                f.write(json.dumps(item) + "\n")

        print(f"Saved {len(pairs)} training pairs to {output_path}")

    except Exception as e:
        print(f"[ERROR] Failed to save training pairs: {e}")
        raise


def preprocess_dataset(data_dir: str = "data/kaggle", output_path: str = "data/processed/train_pairs.jsonl") -> None:
    """Main entry point for preprocessing the StackSample dataset."""
    try:
        # Skip preprocessing if a valid JSONL file already exists
        if os.path.exists(output_path):
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        json.loads(first_line)  # validate JSON
                        print(f"Found existing processed file at {output_path}. Skipping preprocessing.")
                        return
            except Exception:
                print("Existing file is invalid or empty. Rebuilding...")

        print("Loading raw data...")
        questions, answers = load_raw_data(data_dir)

        print("Building training pairs...")
        pairs = build_training_pairs(questions, answers)

        print("Saving processed dataset...")
        save_pairs(pairs, output_path)

    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    data_dir = "data/kaggle"
    preprocess_dataset(data_dir)
