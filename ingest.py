import json
import re
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

# Sections we never want in the index (compared after normalizeHeading)
SKIP_SECTIONS = frozenset({"references", "acknowledgments", "acknowledgements"})

# HTML tags like <sup>, and Markdown marks * _ `
MARKUP = re.compile(r"<[^>]+>|[*_`]")
HEADING = re.compile(r"^#{1,6}\s")
# A 4-digit year like 1998 or 2020, used to recognize bibliography entries
YEAR = re.compile(r"\b(19|20)\d{2}\b")

###########################################################

def stopwordRatio(text):
    """Share of words that are common English stopwords (0.0 - 1.0)."""
    words = text.lower().split()
    return sum(w in STOPWORDS for w in words) / len(words) if words else 0.0

def isReferenceList(chunk):
    """True if most lines look like bibliography entries ('- ...' with a year)."""
    lines = [l for l in chunk.splitlines() if l.strip()]
    refLike = sum(l.lstrip().startswith("- ") and bool(YEAR.search(l)) for l in lines)
    return bool(lines) and refLike / len(lines) > 0.5

def isClean(chunk):
    """Keep long, real-English chunks that are not reference lists."""
    return (len(chunk) >= MIN_CHARS
            and stopwordRatio(chunk) >= MIN_STOPWORD_RATIO
            and not isReferenceList(chunk))

def normalizeHeading(line):
    """'### **References**' and '## REFERENCES' both become 'references'."""
    txt = MARKUP.sub("",line.lstrip("#"))         # drop leading #'s and markup
    return " ".join(txt.split()).lower()          # collapse spaces, lowercase

def splitSections(txt,section):
    """Split one page into (section , text) parts; return the section the page ends in."""
    parts, lines = [],[]
    for line in txt.splitlines():
        if HEADING.match(line):             # a heading starts a new section
            if lines:
                parts.append((section, "\n".join(lines)))
            lines = []
            section = normalizeHeading(line)
        lines.append(line)                   # heading stays with its own content
    if lines:
        parts.append((section,"\n".join(lines)))
    return parts,section

def chunksFromPage(pdf, page, section):
    """Split one page into clean chunk records. Skip for unnecessary sections."""
    pageNo = page["metadata"]["page_number"]                    # starts at 1, used for citations
    text = unicodedata.normalize("NFKC", page["text"])          # safety net: ﬁ -> fi
    parts, section = splitSections(text,section)
 
    records = [
        {"source": pdf.name, "page": pageNo, "section": sec, "text": chunk}
        for sec, body in parts if sec not in SKIP_SECTIONS    # drop whole unwanted sections
        for chunk in splitter.split_text(body) if isClean(chunk)
    ]

    for n, r in enumerate(records):
        r["id"] = f"{pdf.stem}-p{pageNo}-c{n}"               # e.g. react-p3-c1
    return records, section


def buildChunks():
    """Read every PDF page by page and collect all clean chunks."""
    records = []
    for pdf in sorted(DOCS_DIR.glob("*.pdf")):       # sorted -> same order every run  
        section = ""                 
        for page in pymupdf4llm.to_markdown(str(pdf), page_chunks=True):  # one dict per page, as Markdown
            pageRecords, section = chunksFromPage(pdf, page, section)
            records.extend(pageRecords)                                   # add this page's chunks
    return records


###########################################################

if __name__ == "__main__":
    records = buildChunks()
    OUT_FILE.parent.mkdir(exist_ok=True)                                # create chunks/ if missing
    lines = (json.dumps(r, ensure_ascii=False) for r in records)        # keep non-ASCII characters
    OUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")      # rebuild the file each run
    print(len(records), "chunks saved to", OUT_FILE)