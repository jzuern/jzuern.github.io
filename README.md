# Run locally
bundle exec jekyll build
bundle exec jekyll serve


# Repo branches
- gh-pages: rake publish publishes to this branch. Is used to produce jzuer.github.io site
- deploy: 
- master: stale (do not use)


# Publish (to github pages) with:
rake site:publish



# To remember

## Image resizing

`convert trackletmapper.png -resize 200x trackletmapper-200.png`

## Bibtex

abbr: Adds an abbreviation to the left of the entry. You can add links to these by creating a venue.yaml-file in the _data folder and adding entries that match.
abstract: Adds an "Abs" button that expands a hidden text field when clicked to show the abstract text
arxiv: Adds a link to the Arxiv website (Note: only add the arxiv identifier here - the link is generated automatically)
bibtex_show: Adds a "Bib" button that expands a hidden text field with the full bibliography entry
html: Inserts a "HTML" button redirecting to the user-specified link
pdf: Adds a "PDF" button redirecting to a specified file (if a full link is not specified, the file will be assumed to be placed in the /assets/pdf/ directory)
supp: Adds a "Supp" button to a specified file (if a full link is not specified, the file will be assumed to be placed in the /assets/pdf/ directory)
blog: Adds a "Blog" button redirecting to the specified link
code: Adds a "Code" button redirecting to the specified link
poster: Adds a "Poster" button redirecting to a specified file (if a full link is not specified, the file will be assumed to be placed in the /assets/pdf/ directory)
slides: Adds a "Slides" button redirecting to a specified file (if a full link is not specified, the file will be assumed to be placed in the /assets/pdf/ directory)
website: Adds a "Website" button redirecting to the specified link

