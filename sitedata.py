"""Load structured site data (publications from BibTeX, news from YAML) for Pelican."""

from __future__ import annotations

import re
from pathlib import Path

import bibtexparser
import yaml
from bibtexparser.customization import convert_to_unicode

DATA_DIR = Path(__file__).parent / "data"

_LATEX_CLEANUP = re.compile(r"[{}]")


def _clean(value: str) -> str:
    return _LATEX_CLEANUP.sub("", " ".join(value.split()))


def _format_authors(raw: str, highlight: str = "Zürn") -> list[dict[str, str | bool]]:
    """Convert a BibTeX author field into a list of {name, is_me} dicts."""
    authors = []
    for author in raw.split(" and "):
        author = _clean(author).strip()
        if "," in author:
            last, first = [part.strip() for part in author.split(",", 1)]
            name = f"{first} {last}"
        else:
            name = author
        authors.append({"name": name, "is_me": highlight in name})
    return authors


def _venue(entry: dict[str, str]) -> str:
    if "journal" in entry:
        return _clean(entry["journal"])
    if "booktitle" in entry:
        return _clean(entry["booktitle"])
    if entry.get("ENTRYTYPE") == "phdthesis":
        return f"PhD thesis, {_clean(entry.get('school', ''))}"
    return "Preprint"


def _raw_bibtex(entry: dict[str, str]) -> str:
    """Reconstruct a minimal BibTeX string for the copy-to-clipboard button."""
    skip = {
        "ID", "ENTRYTYPE", "pdf", "website", "code", "teaser", "selected",
        "bibtex_show", "abbr", "arxiv", "supp", "poster", "slides", "blog", "award",
    }
    lines = [f"@{entry['ENTRYTYPE']}{{{entry['ID']}," ]
    for key, value in entry.items():
        if key in skip:
            continue
        lines.append(f"  {key} = {{{value}}},")
    lines.append("}")
    return "\n".join(lines)


def load_publications() -> list[dict[str, object]]:
    """Parse data/papers.bib into template-ready publication dicts, newest first."""
    bib_path = DATA_DIR / "papers.bib"
    parser = bibtexparser.bparser.BibTexParser(common_strings=True)
    parser.customization = convert_to_unicode
    with bib_path.open(encoding="utf-8") as handle:
        database = bibtexparser.load(handle, parser=parser)

    publications = []
    for entry in database.entries:
        publications.append(
            {
                "key": entry["ID"],
                "title": _clean(entry["title"]),
                "authors": _format_authors(raw=entry["author"]),
                "venue": _venue(entry),
                "year": int(entry["year"]),
                "award": entry.get("award", ""),
                "pdf": entry.get("pdf", ""),
                "website": entry.get("website", ""),
                "code": entry.get("code", ""),
                "arxiv": entry.get("arxiv", ""),
                "teaser": entry.get("teaser", ""),
                "selected": entry.get("selected", "").lower() == "true",
                "bibtex": _raw_bibtex(entry),
            }
        )
    publications.sort(key=lambda pub: pub["year"], reverse=True)
    return publications


_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def load_news() -> list[dict[str, str]]:
    """Load data/news.yml entries ({date, text}), newest first, with rendered HTML."""
    news_path = DATA_DIR / "news.yml"
    with news_path.open(encoding="utf-8") as handle:
        items = yaml.safe_load(handle) or []
    items.sort(key=lambda item: str(item["date"]), reverse=True)
    for item in items:
        item["date_str"] = item["date"].strftime("%b %Y")
        item["html"] = _MD_LINK.sub(
            r'<a href="\2" target="_blank" rel="noopener">\1</a>', item["text"].strip()
        )
    return items
