# RiskScoop AI

**System analizy ryzyka powodziowego dla infrastruktury krytycznej z wykorzystaniem agenta AI**

Praca inzynierska — Politechnika Lodzka, FTIMS, 2025
Autor: Nikodem Wojtczak

---

## Struktura repozytorium

```
.
├── riskscoop-ai/          # Aplikacja webowa
│   └── backend/           # Backend (Python, FastAPI, Agno AgentOS)
│       ├── my_os.py               # Konfiguracja agenta i serwer FastAPI
│       ├── tools/                 # Narzedzia agenta AI (6 narzedzi)
│       │   ├── search_division_tool.py    # Wyszukiwanie lokalizacji (Nominatim)
│       │   ├── get_division_tool.py       # Pobieranie granic administracyjnych (Overpass API)
│       │   ├── get_overture_tool.py       # Zapytania SQL do Overture Maps (MotherDuck)
│       │   ├── get_flood_tool.py          # Dane powodziowe z Copernicus GloFAS
│       │   └── intersect_flood_tool.py    # Analiza przeciec przestrzennych (GeoPandas)
│       ├── services/              # Serwisy backendowe
│       │   ├── nominatim_service.py       # Integracja z OSM + algorytm assemble_rings
│       │   ├── md_service.py              # DatabaseEngine singleton (MotherDuck/DuckDB)
│       │   ├── glofas_service.py          # Konwersja GeoTIFF na GeoDataFrame
│       │   ├── layer_service.py           # Zarzadzanie warstwami GeoJSON (UUID)
│       │   ├── sql_tool_utils.py          # Filtrowanie przestrzenne (bbox + intersection)
│       │   ├── copernicus_glofas.py       # Pobieranie danych z Copernicus CEMS
│       │   ├── raster_utils.py            # Narzedzia do przetwarzania rastrow
│       │   ├── type_converter.py          # Konwersja typow danych
│       │   └── overture/                  # Schematy tabel Overture Maps
│       │       ├── buildings.py
│       │       ├── infrastructure.py
│       │       ├── places.py
│       │       └── transportation.py
│       ├── test_frontend/         # Prosty frontend testowy (HTML)
│       ├── Dockerfile             # Obraz Docker
│       ├── requirements.txt       # Zaleznosci Python
│       ├── .env.example           # Szablon zmiennych srodowiskowych
│       └── .env.docker            # Konfiguracja dla Dockera
│
├── thesis/                # Praca inzynierska (LaTeX)
│   ├── main.tex                   # Dokument glowny
│   ├── main.pdf                   # Skompilowany PDF
│   ├── streszczenieislowakluczowe.tex  # Streszczenie i slowa kluczowe
│   ├── wstep.tex                  # Wstep
│   ├── cel_i_zakres.tex           # Cel i zakres pracy
│   ├── rozdzial1.tex              # Rozdzial 1: Przeglad literatury i analiza istniejacych rozwiazan
│   ├── rozdzial3.tex              # Rozdzial 2: Projekt systemu
│   ├── rozdzial4.tex              # Rozdzial 3: Implementacja
│   ├── rozdzial5.tex              # Rozdzial 4: Testy i walidacja
│   ├── podsumowanie.tex           # Podsumowanie i wnioski
│   ├── wykazy.tex                 # Wykaz skrotow, definicji, zalacznikow
│   ├── literatura.bib             # Bibliografia (BibTeX)
│   ├── titlepage.tex              # Strona tytulowa
│   └── Figures/                   # Zrzuty ekranu z analizy agenta
│       ├── Szpitale_w_Genewie.png
│       ├── Szkoly_w_Warszawie.png
│       ├── Budynki_w_Krakowie.png
│       ├── Infrastruktura_w_Plocku.png
│       └── Infrastruktura_we_Wroclawiu.png
│
└── temat-pracy.txt        # Temat pracy inzynierskiej
```

## Technologie

| Komponent | Technologia |
|-----------|-------------|
| Agent AI | Agno Framework (AgentOS) + Google Gemini |
| Backend | Python 3.11+, FastAPI |
| Baza danych | MotherDuck (DuckDB w chmurze) |
| Dane infrastrukturalne | Overture Maps Foundation (2,6 mld budynkow) |
| Dane powodziowe | Copernicus CEMS GloFAS |
| Geokodowanie | OpenStreetMap Nominatim + Overpass API |
| Analiza przestrzenna | GeoPandas, Shapely |
| Wizualizacja map | Mapbox GL JS |
| Praca dyplomowa | LaTeX (pdflatex + bibtex) |

## Uruchomienie backendu

### Wymagane klucze API

Aplikacja wymaga kluczy API do zewnetrznych uslug. Skopiuj `.env.example` do `.env` i uzupelnij:

```bash
cd riskscoop-ai/backend
cp .env.example .env
```

Wymagane klucze:
- `GOOGLE_API_KEY` — Google Gemini API (model agenta)
- `MOTHERDUCK_TOKEN` — MotherDuck (baza Overture Maps)
- `MAPBOX_TOKEN` — Mapbox (wizualizacja map)
- `GLOFAS_API_KEY` — Copernicus CDS (dane powodziowe)

### Uruchomienie z Dockerem

```bash
cd riskscoop-ai/backend
docker build -t riskscoop-ai .
docker run -p 7777:8000 --env-file .env riskscoop-ai
```

Serwer dostepny na `http://localhost:7777`

### Uruchomienie lokalne

```bash
cd riskscoop-ai/backend
pip install -r requirements.txt
python my_os.py
```

Serwer dostepny na `http://localhost:7777`

## Przykladowe zapytania do agenta

- *"Find hospitals at flood risk in Geneva, Switzerland"*
- *"Which schools in Warsaw are at high flood risk?"*
- *"Show all buildings at flood risk in Krakow"*
- *"Find critical infrastructure at flood risk in Plock, Poland"*
- *"Show infrastructure at flood risk in Wroclaw, Poland"*

## Kompilacja pracy LaTeX

```bash
cd thesis
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Wynik: `thesis/main.pdf`

## Licencja

Praca inzynierska © 2025 Nikodem Wojtczak, Politechnika Lodzka

Dane zewnetrzne:
- Overture Maps Foundation: [CDLA Permissive 2.0](https://cdla.dev/permissive-2-0/)
- Copernicus GloFAS: [Copernicus License](https://cds.climate.copernicus.eu/)
- OpenStreetMap: [ODbL](https://www.openstreetmap.org/copyright)
