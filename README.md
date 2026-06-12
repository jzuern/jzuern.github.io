# jzuern.github.io

Personal academic website, built with [Pelican](https://getpelican.com) (Python) and a custom theme.

## Branches

- `deploy` — source branch. Push here; GitHub Actions builds the site and publishes it.
- `gh-pages` — build output, served by GitHub Pages. Never edit manually.
- `master` — stale (do not use).

## Local development

```bash
uv sync                                       # install dependencies
uv run pelican content -s pelicanconf.py     # build into output/
uv run pelican --listen --autoreload         # serve at http://localhost:8000 with live reload
```

## Publishing

Push to the `deploy` branch. The GitHub Actions workflow (`.github/workflows/deploy.yml`)
builds the site with `publishconf.py` and force-pushes the result to `gh-pages`.

## Project layout

| Path | Purpose |
|---|---|
| `content/pages/` | About, Projects, Teaching pages (Markdown) |
| `content/blog/` | Blog posts (Markdown, math via `$...$` / `$$...$$`) |
| `content/images/` | All images (referenced as `/images/...`) |
| `content/pdf/` | CV and other documents |
| `data/papers.bib` | Publications database (BibTeX) |
| `data/news.yml` | News items shown on the homepage |
| `theme/jzuern/` | Custom theme (Jinja2 templates + CSS + JS) |
| `sitedata.py` | Parses papers.bib / news.yml into template context |

## Adding content

- **Publication:** add a BibTeX entry to `data/papers.bib`. Supported extra fields:
  `pdf`, `arxiv` (identifier only), `website`, `code`, `teaser` (file in
  `content/images/paper_teasers/`), `selected` (`true` shows it on the homepage), `award`.
- **News item:** add a `{date, text}` entry to `data/news.yml` (Markdown links supported).
- **Blog post:** add `content/blog/YYYY-MM-DD-slug.md` with `Title:`, `Date:`, `Slug:` metadata.
