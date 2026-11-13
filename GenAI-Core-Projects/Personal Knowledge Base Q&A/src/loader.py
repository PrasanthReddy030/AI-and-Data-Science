"""Reads raw documents (.txt, .md, .pdf) from a directory into plain record dicts."""

from pathlib import Path
from pypdf import PdfReader


def load_txt_or_md(dirpath: Path) -> list[dict]:
    """
    Load every .txt and .md file directly inside dirpath.

    Returns one record per file:
        {"doc_id": filename without extension, "source": full path,
         "text": file contents, "page": None}

    Files that fail to decode are skipped and logged, not raised.
    """
    records = []
    dirpath = Path(dirpath)

    for file in sorted(dirpath.iterdir()):
        if file.is_file() and file.suffix.lower() in {".txt", ".md"}:
            try:
                text = file.read_text(encoding="utf-8")

                records.append({
                    "doc_id": file.stem,
                    "source": str(file),
                    "text": text,
                    "page": None,
                })

            except (UnicodeDecodeError, OSError) as e:
                print(f"Skipping {file}: {e}")

    return records


def load_pdfs(dirpath: Path) -> list[dict]:
    """
    Load every .pdf file directly inside dirpath.

    Returns one record per non-empty page across all PDFs:
        {"doc_id": filename without extension, "source": full path,
         "text": page text, "page": 1-indexed page number}

    Corrupt or unreadable PDFs are skipped and logged, not raised.
    """
    records = []
    dirpath = Path(dirpath)

    for pdf_file in sorted(dirpath.iterdir()):
        if pdf_file.is_file() and pdf_file.suffix.lower() == ".pdf":
            try:
                reader = PdfReader(pdf_file)

                for page_num, page in enumerate(reader.pages, start=1):
                    text = (page.extract_text() or "").strip()

                    if not text:
                        continue

                    records.append({
                        "doc_id": pdf_file.stem,
                        "source": str(pdf_file),
                        "text": text,
                        "page": page_num,
                    })

            except Exception as e:
                print(f"Skipping unreadable PDF {pdf_file}: {e}")

    return records