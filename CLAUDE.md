# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains two main components:
1. **RiskScoop AI** (`riskscoop-ai/`) - A flood risk analysis system using AI agents
2. **Engineering Thesis** (`formatka_inz_ftims_2024 2/`) - LaTeX thesis for Lodz University of Technology (FTIMS)

## RiskScoop AI Backend

### Running the Agent

```bash
cd riskscoop-ai/backend
pip install -r requirements.txt
python my_os.py
# Server runs on http://localhost:7777
```

### Environment Variables

Required in `riskscoop-ai/backend/.env`:
- `GOOGLE_API_KEY` - Google Gemini API key
- `MOTHERDUCK_TOKEN` - MotherDuck cloud database token
- `MAPBOX_TOKEN` - Mapbox API token

### Architecture

The backend uses **AgentOS** pattern from Agno framework with 6 tools:

```
my_os.py                    # Main AgentOS configuration (Gemini model)
├── tools/
│   ├── search_division_tool.py   # Nominatim place search
│   ├── get_division_tool.py      # OSM boundary retrieval (Overpass API)
│   ├── get_overture_tool.py      # MotherDuck SQL queries
│   ├── get_flood_tool.py         # Copernicus GloFAS data
│   └── intersect_flood_tool.py   # Spatial intersection analysis
└── services/
    ├── nominatim_service.py      # OSM integration + assemble_rings algorithm
    ├── md_service.py             # DatabaseEngine singleton (thread-local)
    ├── glofas_service.py         # GeoTIFF to GeoDataFrame conversion
    ├── layer_service.py          # GeoJSON file management with UUIDs
    └── sql_tool_utils.py         # Spatial filtering with bounding box
```

### Key Patterns

- **DatabaseEngine Singleton**: Thread-local connections in `md_service.py` for parallel query execution
- **assemble_rings Algorithm**: Reconstructs polygons from OSM way fragments in `nominatim_service.py`
- **Two-stage Spatial Filtering**: Bounding box filter in SQL + precise intersection in Python

### Overture Maps Tables (MotherDuck)

- `overture_buildings` - Buildings with class, height, geometry
- `overture_places` - POIs with primary_category
- `overture_infrastructure` - Infrastructure (bridges, power lines)
- `overture_transportation` - Roads and railways

## LaTeX Thesis

### Building the Document

```bash
cd "formatka_inz_ftims_2024 2"
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

### Thesis Structure

- `main.tex` - Master document with packages and includes
- `rozdzial1.tex` - Literature review
- `rozdzial2.tex` - Technology analysis
- `rozdzial3.tex` - System design (architecture, tools, data model)
- `rozdzial4.tex` - Implementation details
- `podsumowanie.tex` - Conclusions
- `wykazy.tex` - Abbreviations, definitions, attachments list
- `literatura.bib` - BibTeX bibliography

### Formatting Rules

- Use `~` for non-breaking spaces after single-letter Polish prepositions (w, o, i, z, u, a)
- Figures: `Rys. X.Y` caption format, source required (citation or "opracowanie własne")
- Tables: Caption above table
- No code listings in chapters - use attachments (Załącznik A)
