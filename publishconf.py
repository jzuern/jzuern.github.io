"""Production overrides used by CI when publishing to GitHub Pages."""

import os
import sys

sys.path.append(os.curdir)
from pelicanconf import *  # noqa: F401,F403

SITEURL = "https://jzuern.github.io"
RELATIVE_URLS = False

FEED_ALL_ATOM = "feed.xml"
FEED_DOMAIN = SITEURL

DELETE_OUTPUT_DIRECTORY = True
