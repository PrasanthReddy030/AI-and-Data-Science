
from pathlib import Path

from src.loader import load_txt_or_md, load_pdfs


def test_load_txt_or_md_skips_unreadable_file(tmp_path):
    """Unreadable UTF-8 files should be skipped without crashing."""

    # Create one valid text file
    good_file = tmp_path / "good.txt"
    good_file.write_text("This is valid text.", encoding="utf-8")

    # Create one file containing invalid UTF-8 bytes
    bad_file = tmp_path / "bad.txt"
    bad_file.write_bytes(b"\xff\xfe\xfa\xff")

    records = load_txt_or_md(tmp_path)

    # Good file should be loaded
    assert len(records) == 1
    assert records[0]["doc_id"] == "good"
    assert records[0]["text"] == "This is valid text."

    # Bad file should not appear
    assert all(record["doc_id"] != "bad" for record in records)


def test_load_txt_or_md_ignores_unsupported_extensions(tmp_path):
    """Only TXT and Markdown files should be loaded."""

    (tmp_path / "notes.txt").write_text(
        "Some notes.", encoding="utf-8"
    )

    (tmp_path / "data.json").write_text(
        '{"key": "value"}', encoding="utf-8"
    )

    (tmp_path / "image.png").write_bytes(
        b"fake image data"
    )

    records = load_txt_or_md(tmp_path)

    assert len(records) == 1
    assert records[0]["doc_id"] == "notes"


def test_load_txt_or_md_record_shape(tmp_path):
    """Verify the structure of a loaded Markdown record."""

    md_file = tmp_path / "example.md"
    md_file.write_text(
        "# Example\n\nThis is a Markdown document.",
        encoding="utf-8",
    )

    records = load_txt_or_md(tmp_path)

    assert len(records) == 1

    record = records[0]

    # Check exact record keys
    assert set(record.keys()) == {
        "doc_id",
        "source",
        "text",
        "page",
    }

    assert record["doc_id"] == "example"
    assert record["source"] == str(md_file)
    assert record["text"] == (
        "# Example\n\nThis is a Markdown document."
    )
    assert record["page"] is None


def test_load_pdfs_skips_corrupt_pdf(tmp_path):
    """Corrupt PDFs should be skipped without crashing."""

    # Create a file with a PDF extension but invalid contents
    bad_pdf = tmp_path / "corrupt.pdf"
    bad_pdf.write_bytes(b"This is not a valid PDF file.")

    # The loader should catch the error and return no records
    records = load_pdfs(tmp_path)

    assert records == []


def test_load_pdfs_skips_empty_pages(tmp_path, monkeypatch):
    """
    Verify that PDF pages with no extractable text are skipped.

    Mock PdfReader so no real PDF fixture is required.
    """

    import src.loader as loader

    # Create a placeholder PDF file
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"placeholder")

    # Fake PDF page objects
    class FakePage:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    # Simulate a three-page PDF:
    # Page 1: valid text
    # Page 2: empty text
    # Page 3: whitespace only
    class FakePdfReader:
        def __init__(self, filepath):
            self.pages = [
                FakePage("First page content."),
                FakePage(None),
                FakePage("   "),
            ]

    # Replace the real PdfReader with our fake implementation
    monkeypatch.setattr(
        loader,
        "PdfReader",
        FakePdfReader,
    )

    records = load_pdfs(tmp_path)

    # Only the first page should produce a record
    assert len(records) == 1

    assert records[0]["doc_id"] == "sample"
    assert records[0]["text"] == "First page content."
    assert records[0]["page"] == 1
