"""Pelican configuration for jzuern.github.io."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sitedata import load_news, load_publications

AUTHOR = "Jannik Zürn"
SITENAME = "Jannik Zürn"
SITESUBTITLE = "Senior Machine Learning Engineer @ Parallel Domain"
SITEURL = ""

PATH = "content"
TIMEZONE = "Europe/Berlin"
DEFAULT_LANG = "en"
DEFAULT_DATE_FORMAT = "%b %d, %Y"

THEME = "theme/jzuern"

# Content layout
ARTICLE_PATHS = ["blog"]
PAGE_PATHS = ["pages"]
STATIC_PATHS = ["images", "pdf", "extra"]
EXTRA_PATH_METADATA = {
    "extra/favicon.ico": {"path": "favicon.ico"},
    "extra/robots.txt": {"path": "robots.txt"},
}

# URL structure (preserves old al-folio permalinks /blog/<year>/<slug>/)
ARTICLE_URL = "blog/{date:%Y}/{slug}/"
ARTICLE_SAVE_AS = "blog/{date:%Y}/{slug}/index.html"
PAGE_URL = "{slug}/"
PAGE_SAVE_AS = "{slug}/index.html"
ARCHIVES_SAVE_AS = ""
AUTHORS_SAVE_AS = ""
CATEGORIES_SAVE_AS = ""
TAGS_SAVE_AS = ""
AUTHOR_SAVE_AS = ""
CATEGORY_SAVE_AS = ""
TAG_SAVE_AS = ""

# Direct templates: homepage, blog index, publications
DIRECT_TEMPLATES = ["home", "blog", "publications"]
HOME_SAVE_AS = "index.html"
BLOG_SAVE_AS = "blog/index.html"
PUBLICATIONS_SAVE_AS = "publications/index.html"

# Markdown
MARKDOWN = {
    "extension_configs": {
        "markdown.extensions.toc": {},
        "markdown.extensions.tables": {},
        "markdown.extensions.attr_list": {},
        "pymdownx.superfences": {},
        "pymdownx.highlight": {"css_class": "highlight"},
        "pymdownx.arithmatex": {"generic": True},
    },
    "output_format": "html5",
}

# Data injected into all templates
PUBLICATIONS = load_publications()
NEWS = load_news()

SOCIAL_LINKS = {
    "email": "mailto:zuern@informatik.uni-freiburg.de",
    "scholar": "https://scholar.google.com/citations?user=gB9JqUcAAAAJ",
    "github": "https://github.com/jzuern",
    "linkedin": "https://www.linkedin.com/in/jannik-zuern",
    "orcid": "https://orcid.org/0000-0001-9516-905X",
    "medium": "https://medium.com/@jannik-zuern",
}
CV_PATH = "pdf/CV_Jannik_Zürn.pdf"

FEED_ALL_ATOM = None
FEED_ATOM = None
FEED_RSS = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

DEFAULT_PAGINATION = False
RELATIVE_URLS = True
