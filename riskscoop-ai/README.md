# RiskScoop AI

**System analizy ryzyka powodziowego dla infrastruktury krytycznej z wykorzystaniem agenta AI**

Praca inżynierska - Politechnika Łódzka, FTIMS
Autor: Nikodem Wojtczak

## Opis projektu

RiskScoop AI to innowacyjny system wykorzystujący agenta AI do analizy ryzyka powodziowego dla infrastruktury krytycznej w Polsce. System umożliwia zadawanie pytań w języku naturalnym i otrzymywanie szczegółowych analiz ryzyka wraz z rekomendacjami.

### Kluczowe funkcje:
- 🤖 **Agent AI** - przetwarzanie zapytań w języku naturalnym (Claude API)
- 🗺️ **Wizualizacja mapy** - interaktywna mapa z Mapbox GL JS
- 🌊 **Dane powodziowe** - integracja z Copernicus GloFAS
- 🏗️ **Infrastruktura** - dane z Overture Maps Foundation (2.6B budynków)
- 📊 **Analiza ryzyka** - model: Risk = sqrt(Hazard × Vulnerability)

## Architektura

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Frontend     │────▶│    Backend      │────▶│   External      │
│  React + Vite   │     │    FastAPI      │     │   Services      │
│  Mapbox GL JS   │     │    Python       │     │   - Claude API  │
└─────────────────┘     └─────────────────┘     │   - GloFAS      │
                              │                 │   - Overture    │
                              ▼                 └─────────────────┘
                        ┌─────────────────┐
                        │   AI Agent      │
                        │  - NLP Parser   │
                        │  - Risk Calc    │
                        │  - Summarizer   │
                        └─────────────────┘
```

## Instalacja

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt

# Konfiguracja
cp .env.example .env
# Edytuj .env i dodaj klucze API

# Uruchomienie
uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install

# Konfiguracja
cp .env.example .env
# Edytuj .env i dodaj token Mapbox

# Uruchomienie
npm run dev
```

## Użycie

1. Uruchom backend na `http://localhost:8000`
2. Uruchom frontend na `http://localhost:5173`
3. Wpisz zapytanie, np.:
   - "Pokaż szpitale w Warszawie zagrożone powodzią"
   - "Analiza ryzyka dla szkół w Krakowie"
   - "Mosty w Gdańsku z wysokim ryzykiem"

## API Endpoints

| Endpoint | Metoda | Opis |
|----------|--------|------|
| `/health` | GET | Status serwera |
| `/api/query` | POST | Zapytanie do agenta AI |
| `/api/infrastructure` | GET | Pobierz infrastrukturę |
| `/api/flood-risk` | GET | Dane o ryzyku powodziowym |
| `/api/categories` | GET | Dostępne kategorie |

## Model ryzyka

```
Risk = √(Hazard × Vulnerability)
```

Gdzie:
- **Hazard** - prawdopodobieństwo powodzi z GloFAS (0-1)
- **Vulnerability** - współczynnik podatności kategorii:
  - Szpitale: 0.95
  - Oczyszczalnie: 0.90
  - Elektrownie: 0.85
  - Szkoły: 0.70
  - Mosty: 0.60
  - Drogi: 0.40

## Technologie

### Backend
- Python 3.11+
- FastAPI
- Anthropic Claude API
- DuckDB + GeoParquet
- httpx (async HTTP)

### Frontend
- React 18
- Vite
- Mapbox GL JS
- Axios

### Dane
- Copernicus GloFAS (prognoza powodzi)
- Overture Maps Foundation (infrastruktura)

## Licencja

Praca inżynierska © 2024 Nikodem Wojtczak

Dane zewnętrzne:
- Overture Maps: [CDLA Permissive 2.0](https://overturemaps.org/download/)
- Copernicus GloFAS: [Copernicus License](https://cds.climate.copernicus.eu/)
