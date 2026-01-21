#!/usr/bin/env python3
import json
import os
import time
import logging
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from difflib import SequenceMatcher

# Configurable via env
API_URL = os.getenv("API_URL", "http://localhost:8000/graphql")
SESSION_ID = os.getenv("SESSION_ID", "eval-session")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "10"))  # seconds
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))
RETRIES = int(os.getenv("RETRIES", "2"))
BACKOFF_FACTOR = float(os.getenv("BACKOFF_FACTOR", "0.5"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def lexical_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()


def post_graphql(payload: Dict[str, Any]) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    last_exc = None
    for attempt in range(RETRIES + 1):
        try:
            resp = requests.post(API_URL, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            else:
                logging.warning("Non-200 response %s: %s", resp.status_code, resp.text[:200])
        except requests.RequestException as e:
            last_exc = e
            logging.warning("Request failed (attempt %d/%d): %s", attempt + 1, RETRIES + 1, e)
        time.sleep(BACKOFF_FACTOR * (2 ** attempt))
    raise RuntimeError(f"GraphQL request failed after retries: {last_exc}")


def ask_backend(question: str, session_id: str) -> Dict[str, Any]:
    query = """
      mutation Chat($sessionId: String!, $message: String!) {
        chat(sessionId: $sessionId, message: $message) {
          reply
          sources {
            id
            score
            text
          }
        }
      }
    """
    payload = {"query": query, "variables": {"sessionId": session_id, "message": question}}
    data = post_graphql(payload)
    # Defensive extraction
    chat = data.get("data", {}).get("chat")
    if not chat:
        raise ValueError(f"Unexpected GraphQL response shape: {data}")
    return chat


def evaluate_one(q: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"id": q.get("id"), "question": q.get("question")}
    try:
        start = time.time()
        chat = ask_backend(q["question"], SESSION_ID)
        latency = time.time() - start

        answer = chat.get("reply", "")
        sources = chat.get("sources") or []

        correctness = lexical_similarity(answer, q.get("expected_answer", ""))
        faithfulness = lexical_similarity(answer, " ".join([s.get("text", "") for s in sources]))

        result.update({
            "answer": answer,
            "correctness": correctness,
            "faithfulness": faithfulness,
            "latency": latency,
            "num_sources": len(sources),
            "error": None
        })
    except Exception as e:
        logging.exception("Evaluation failed for id %s", q.get("id"))
        result.update({
            "answer": None,
            "correctness": 0.0,
            "faithfulness": 0.0,
            "latency": None,
            "num_sources": 0,
            "error": str(e)
        })
    return result


def evaluate():
    with open("kubernetes_questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    # Warmup single request to avoid cold-start skew
    if questions:
        try:
            ask_backend(questions[0]["question"], SESSION_ID)
            logging.info("Warmup request completed")
        except Exception:
            logging.info("Warmup failed, continuing with evaluation")

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(evaluate_one, q): q for q in questions}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            logging.info("[%s] correctness=%.2f faithfulness=%.2f latency=%s num_sources=%d error=%s",
                         res.get("id"), res.get("correctness"), res.get("faithfulness"),
                         f"{res.get('latency'):.2f}s" if res.get("latency") else "N/A",
                         res.get("num_sources"), res.get("error"))

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info("Evaluation complete. Results saved to results.json")


if __name__ == "__main__":
    evaluate()
