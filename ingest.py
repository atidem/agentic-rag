import json
import unicodedata
from pathlib import Path

import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter

###########################################################
DOCS_DIR = Path("docs")                     # input PDFs
OUT_FILE = Path("chunks/chunks.jsonl")      # one chunk per line (JSON Lines)

MIN_CHARS = 100                             # shorter chunks are page numbers or table leftovers
MIN_STOPWORD_RATIO = 0.05                   # below this: garbled figure text or reference lists

# Most common English words; garbled text has almost none of them
STOPWORDS = frozenset({"the", "of", "and", "to", "a", "in", "is", "that",
                       "for", "we", "on", "with", "as", "by", "are"})

# Splits on paragraphs first, then lines, then spaces, so chunks rarely cut words in half
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

###########################################################

def stopwordRatio(text: str) -> float:
    """Share of words that are common English stopwords (0.0 - 1.0)."""
    words = text.lower().split()
    return sum(w in STOPWORDS for w in words) / len(words) if words else 0.0


def isClean(chunk: str) -> bool:
    """Keep chunks that are long enough and look like real English text."""
    return len(chunk) >= MIN_CHARS and stopwordRatio(chunk) >= MIN_STOPWORD_RATIO


def chunksFromPage(pdf: Path, page: dict) -> list[dict]:
    """Split one page into clean chunk records."""
    pageNo = page["metadata"]["page_number"]                    # starts at 1, used for citations
    text = unicodedata.normalize("NFKC", page["text"])          # safety net: ﬁ -> fi
    return [
        {"id": f"{pdf.stem}-p{pageNo}-c{n}",                    # e.g. react-p3-c1
         "source": pdf.name, "page": pageNo, "text": chunk}
        for n, chunk in enumerate(splitter.split_text(text))   # chunks never cross pages
        if isClean(chunk)
    ]


def buildChunks() -> list[dict]:
    """Read every PDF page by page and collect all clean chunks."""
    records = []
    for pdf in sorted(DOCS_DIR.glob("*.pdf")):                          # sorted -> same order every run
        for page in pymupdf4llm.to_markdown(str(pdf), page_chunks=True):  # one dict per page, as Markdown
            records.extend(chunksFromPage(pdf, page))                   # add this page's chunks
    return records


###########################################################

if __name__ == "__main__":
    records = buildChunks()
    OUT_FILE.parent.mkdir(exist_ok=True)                                # create chunks/ if missing
    lines = (json.dumps(r, ensure_ascii=False) for r in records)        # keep non-ASCII characters
    OUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")      # rebuild the file each run
    print(len(records), "chunks saved to", OUT_FILE)