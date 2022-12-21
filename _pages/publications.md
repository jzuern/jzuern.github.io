---
layout: page
permalink: /publications/
title: publications
description: A complete list of my publications can be found on my <a href='https://scholar.google.com/citations?user=gB9JqUcAAAAJ&hl=en'>Google Scholar</a>.
years: [2023, 2022, 2021, 2020]
nav: true
---
<!-- _pages/publications.md -->
<div class="publications">

{%- for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f papers -q @*[year={{y}}]* %}
{% endfor %}

</div>
