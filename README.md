# RegLens

GCAA "Innovation Takes Flight" · RegLens challenge (Safety sector).

RegLens reads the Civil Aviation Regulations (CARs), splits them into articles, and finds four kinds of problems: conflicting values, duplicate or divergent definitions, broken references, and duplicate or overlapping requirements. Each finding cites the regulation and article number, quotes the source text verbatim, and recommends an action. An expert reviews every finding. Official text is never modified.

## Run

```bash
pip install -r requirements.txt
python ingest.py data/raw/*          # regulations -> data/articles.json
python run_pipeline.py               # articles -> data/findings.json (+ references, similarity, terms)
python run_pipeline.py --mock        # same, on the fictional test data
python evaluate.py                   # recall vs data/expert_labels.csv
streamlit run app.py                 # expert review screen
```

## Who owns what

Each person commits only their own files. The shared formats and function signatures are in [CONTRACT.md](CONTRACT.md).

| Person | Files |
|---|---|
| P1 · Ingest | `ingest.py`, `run_pipeline.py`, `evaluate.py`, `requirements.txt` |
| P2 · Rules | `detect_rules.py` |
| P3 · Similarity | `detect_similarity.py` |
| P4 · App | `app.py` |
| P5 · AI | `ai_assist.py`, `prompts/` |

Test data: `data/mock_articles.json` and `data/mock_findings.json` (fictional, for testing only).

## Real data (Phase 2: CAR I + CAR IX)

Download from https://www.gcaa.gov.ae/en/epublication/civil-aviation-regulations/ into `data/raw/`, keeping the original file names:

| RegLens name | GCAA file | Folder |
|---|---|---|
| `CAR-DEF` | CAR-DEF – DEFINITIONS – ISSUE 02.pdf | CAR I - DEFINITIONS and BASIC REGULATION |
| `CAR-BR` | CAR-BR - ISSUE 02.pdf | CAR I - DEFINITIONS and BASIC REGULATION |
| `CAR-ADR` | CAR ADR- AERODROMES - ISSUE 11.pdf | CAR IX - AERODROMES REGULATIONS |
| `CAR-AGA-PART-ADR` | CAR-AGA - PART-ADR - ISSUE 02.pdf | CAR IX → CAR AGA - AERODROMES AND GROUND AIDS |

CAR-DEF is a glossary table; `ingest.py` turns each term into one article (`CAR-DEF|Apron (APN)`). Needs `pdfplumber`.

Optional: `data/regulations.csv` with columns `file,regulation` overrides the regulation name taken from each file name.
