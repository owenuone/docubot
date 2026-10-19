from pathlib import Path

import pytest

from app.ingest import IngestError, load_pages, split_text


def test_split_short_text_is_single_chunk() -> None:
    assert split_text("hello   world\n", size=800, overlap=150) == ["hello world"]


def test_split_empty_text() -> None:
    assert split_text("  \n ", size=800, overlap=150) == []


def test_split_long_text_overlaps() -> None:
    text = "x" * 1000
    chunks = split_text(text, size=800, overlap=150)
    assert [len(c) for c in chunks] == [800, 350]


def test_load_txt(tmp_path: Path) -> None:
    path = tmp_path / "faq.txt"
    path.write_text("Refunds within 30 days.", encoding="utf-8")
    pages = load_pages(path)
    assert len(pages) == 1 and pages[0].number is None


def test_unsupported_suffix(tmp_path: Path) -> None:
    path = tmp_path / "data.xlsx"
    path.write_bytes(b"")
    with pytest.raises(IngestError):
        load_pages(path)
