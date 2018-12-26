---
layout: post
title: "Hello Blog World"
description: "How to setup and use the blog"
date: 2018-12-26
tags: blog
comments: true
---



### Customization
To customize various details - title/description of the website, your SNS accout names, etc - edit the `_config.yml` file. 

### Adding posts
```
rake post title="A Title" [date="2012-02-09"] [tags=[tag1,tag2]] [category="category"]
```
This will create a markdown file in the default folder where all posts are stored in Jekyll; `_post`.

If you wish to **change the directory where posts are saved**, go to the `Rakefile` and edit the `CONFIG = { 'posts': CUSTOM_PATH_HERE }`. This will allow `rake post` to know where to save the new posts to.

The **drafts** you are working on can be saved in the `_drafts` directory. When you push your code to the server, files in this directory will NOT be included to the list o posts.
