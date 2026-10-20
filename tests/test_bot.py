from app.bot import format_sources
from app.rag import Source


def _src(doc: str, page: int | None = None) -> Source:
    return Source(document=doc, page=page, text="", score=1.0)


def test_format_sources_lists_only_cited():
    sources = [_src("faq.md"), _src("manual.pdf", 3), _src("other.md")]
    out = format_sources("Yes [2]. Also [1][2].", sources)
    assert "faq.md" in out and "manual.pdf p.3" in out and "other.md" not in out


def test_format_sources_empty_without_citations():
    assert format_sources("Sorry, I couldn't find that.", [_src("faq.md")]) == ""
