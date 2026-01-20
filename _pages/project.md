---
layout: page
title: Industry
permalink: /project/  
nav: true
nav_order: 3
display_categories: []
horizontal: false
---

<!-- pages/projects.md -->
<div class="projects">
  <p class="description-style">I am passionate about leveraging AI techniques to create impactful, real-world solutions in consumer products. Over seven years in the industry, I served as an R&D Director at <a href="https://www.sensetime.com/en">SenseTime</a>, where I led the Extended Reality Lab and Smart Video Group, working with <a href="https://www.sensetime.com/en/investor_corp_governance">Xiaogang Wang</a> and <a href="https://scholar.google.com/citations?user=AerkT0YAAAAJ&hl=en">Chen Qian</a>.
  
  During this time, I directed the development and research of AI models for two flagship product platforms -- <a href="https://www.sensetime.com/en/product-detail?categoryId=1163&gioNav=1">SenseMARS Agent</a> and <a href="https://www.sensetime.com/en/product-detail?categoryId=32326&gioNav=1">SenseVideo</a>. These platforms collectively powered solutions for over <strong>20 business clients</strong> and reached more than <strong>10 million end users</strong> worldwide.</p>
  {%- if site.enable_project_categories and page.display_categories %}
  <!-- Display categorized projects -->
  {%- for category in page.display_categories %}
  <h2 class="category">{{ category }}</h2>
  {%- assign categorized_projects = site.projects | where: "category", category -%}
  {%- assign sorted_projects = categorized_projects | sort: "importance" %}
  <!-- Generate cards for each project -->
  {% if page.horizontal -%}
  <div class="container">
    <div class="row row-cols-2">
    {%- for project in sorted_projects -%}
      {% include projects_horizontal.html %}
    {%- endfor %}
    </div>
  </div>
  {%- else -%}
  <div class="grid">
    {%- for project in sorted_projects -%}
      {% include projects.html %}
    {%- endfor %}
  </div>
  {%- endif -%}
  {% endfor %}

{%- else -%}
<!-- Display projects without categories -->
  {%- assign sorted_projects = site.projects | sort: "importance" -%}
  <!-- Generate cards for each project -->
  {% if page.horizontal -%}
  <div class="container">
    <div class="row row-cols-2">
    {%- for project in sorted_projects -%}
      {% include projects_horizontal.html %}
    {%- endfor %}
    </div>
  </div>
  {%- else -%}
  <div class="grid">
    {%- for project in sorted_projects -%}
      {% include projects.html %}
    {%- endfor %}
  </div>
  {%- endif -%}
{%- endif -%}
</div>
