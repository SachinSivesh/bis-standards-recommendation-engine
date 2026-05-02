"""
pdf_processor.py
----------------
Extracts BIS standards from dataset.pdf using PyMuPDF.
Each standard block is stored as:
    {
        "standard": "IS 269: 1989",
        "title":    "ORDINARY PORTLAND CEMENT, 33 GRADE",
        "content":  "full text of this standard's section"
    }
"""

import re
import json
import fitz  # PyMuPDF

# Pattern to match IS standard identifiers like:
#   IS 269 : 1989
#   IS 1489 (Part 2) : 1991
#   IS 2185 (Part 1) : 1979
STANDARD_PATTERN = re.compile(
    r'(IS\s+\d+(?:\s*\(Part\s*\d+\))?\s*:\s*\d{4})',
    re.IGNORECASE
)


def normalize_standard_id(raw: str) -> str:
    """Normalize 'IS  269 :  1989' → 'IS 269: 1989'"""
    # Collapse internal spaces around the colon
    s = re.sub(r'\s+', ' ', raw.strip())
    s = re.sub(r'\s*:\s*', ': ', s)
    return s


def extract_text_from_pdf(pdf_path: str) -> str:
    """Return the full concatenated text of the PDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    return "\n".join(pages)


def parse_standards(text: str) -> list[dict]:
    """
    Split the full text into per-standard blocks.
    Returns a list of dicts with keys: standard, title, content.
    """
    # Find all positions of standard headings
    matches = list(STANDARD_PATTERN.finditer(text))

    if not matches:
        raise ValueError("No BIS standards found in PDF text. Check the PDF.")

    standards = []
    for i, match in enumerate(matches):
        std_raw = match.group(0)
        std_id = normalize_standard_id(std_raw)

        # The block runs from this match to the next match (or EOF)
        block_start = match.start()
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block_text = text[block_start:block_end].strip()

        # Extract title: the line immediately after the standard ID
        lines = block_text.splitlines()
        title = ""
        for line in lines[1:6]:  # look within first 5 lines after header
            line = line.strip()
            if line and not STANDARD_PATTERN.match(line):
                title = line
                break

        standards.append({
            "standard": std_id,
            "title":    title,
            "content":  block_text
        })

    return standards


def load_or_extract(pdf_path: str, cache_path: str = "data/standards_cache.json") -> list[dict]:
    """
    Load from cache if available, otherwise extract from PDF and cache.
    """
    import os
    if os.path.exists(cache_path):
        print(f"[PDF Processor] Loading cached standards from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"[PDF Processor] Extracting text from {pdf_path} …")
    text = extract_text_from_pdf(pdf_path)
    print(f"[PDF Processor] Total characters extracted: {len(text):,}")

    standards = parse_standards(text)
    print(f"[PDF Processor] Standards detected: {len(standards)}")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(standards, f, ensure_ascii=False, indent=2)
    print(f"[PDF Processor] Cache saved to {cache_path}")

    return standards


if __name__ == "__main__":
    standards = load_or_extract("data/dataset.pdf")
    for s in standards[:5]:
        print(s["standard"], "|", s["title"][:60])
