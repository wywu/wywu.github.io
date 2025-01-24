---
layout: page
title: Research
permalink: /research/
description: The central goal of my research is to make Artificial Intelligence (AI) systems more and more usable. An AI system includes two critical functions -- 1) perceiving the real world that AI interacts with and 2) expressing the personal characteristic/action and situated environment of AI itself. To ultimately achieve the usability of AI systems, my research focuses on the following two key directions -- 1) learning to perceive the real world to make the perceiving more effective with less supervision and 2) learning to generate/re-create the digital world to make the expressing easier with less human intervention. Further, my research also involves the construction of datasets and open-source software to facilitate the development of academia, centered around these two research directions. Moving forward, I am recently passionate about <em>human-like AI systems</em>., which will have better common sense in their perceiving system, as well as better creativity in their expressing system. This page lists some of my previous work.
nav: false
nav_order: 2
display_categories: [neural rendering, generative model, human generation, human perceiving]
horizontal: false
---

<!-- pages/projects.md -->
<div class="projects">
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



