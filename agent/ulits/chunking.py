"""
Text chunking utilities.
Uses a recursive, code-aware splitter for engineering documents.
"""

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
    """
    Split text into semantically meaningful chunks.
    Prioritizes code blocks, paragraphs, and newlines.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n```",   # end of code block
            "```",     # start of code block
            "\n\n",    # paragraphs
            "\n",      # lines
            ". ",      # sentences
            " ",       # words
            ""         # fallback: character-level
        ],
    )

    return splitter.split_text(text)
