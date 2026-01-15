"""
Utility functions for loading text from different document formats.
Supports PDF, Markdown, and plain text files.
"""

import os
from typing import Optional, List

import markdown
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader


def load_text_from_file(file_path: str, temp_files: Optional[List[str]] = None) -> Optional[str]:
    """
    Load text content from a file based on its extension.
    Optionally accepts a list of temporary files to delete after loading.
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    try:
        if ext == ".pdf":
            return _load_pdf(file_path)

        if ext in [".txt", ".text"]:
            return _load_text(file_path)

        if ext in [".md", ".markdown"]:
            return _load_markdown(file_path)

        return None

    finally:
        # Cleanup temporary files if provided
        if temp_files:
            for t in temp_files:
                if os.path.exists(t):
                    try:
                        os.remove(t)
                    except Exception:
                        pass  # Never break ingestion due to cleanup failure


def _load_pdf(file_path: str) -> Optional[str]:
    try:
        reader = PdfReader(file_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages).strip()
        return text or None
    except Exception:
        return None

def _load_text(file_path: str) -> str:
    """Load plain text from a .txt file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def _load_markdown(file_path: str) -> str:
    """Convert Markdown to plain text."""
    with open(file_path, "r", encoding="utf-8") as f:
        html = markdown.markdown(f.read())
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator="\n")
