# RegLens shared contract (v2)

Do not change without the lead. Extra fields go in `extra`.

```text
=== REGLENS SHARED CONTRACT v2 (do not change; extra fields go in "extra") ===
Project: RegLens, a tool for GCAA (UAE) that detects duplication, overlap and conflict across Civil Aviation Regulations (CARs) and recommends an action per case. It SUPPORTS the expert and never replaces them. It NEVER edits official text. Every result is traceable to regulation + article number.

Language: Python 3.10+. Libraries: pypdf, pdfplumber, python-docx, scikit-learn, pandas, openpyxl, streamlit, anthropic (optional), openai (optional). Text is Arabic and/or English.

Repo layout:
  data/raw/                 original regulation files (pdf/docx/txt)
  data/articles.json        list[Article]       (written by ingest.py)
  data/findings.json        list[Finding]       (written by run_pipeline.py)
  data/reviews.json         {finding_id: Review} (written by app.py)
  data/references.json      list[Reference]     (written by run_pipeline.py via detect_rules.references)
  data/similarity.json      list[[id_a,id_b,score]] (written by run_pipeline.py via detect_similarity.similar_pairs)
  data/terms.json           {term_key: [{article_id, term, text}]} (written by run_pipeline.py via detect_similarity.term_index)
  data/expert_labels.csv    columns: article_a,article_b,type
  data/evaluation.json      written by evaluate.py
  data/mock_articles.json   fixture below (create it if missing)
  data/mock_findings.json   fixture below (create it if missing)
  ingest.py  detect_rules.py  detect_similarity.py  ai_assist.py  evaluate.py  run_pipeline.py  app.py  prompts/

Function signatures (exact):
  ingest.parse_documents(paths: list[str]) -> list[Article]
  detect_rules.detect(articles: list[Article], config: dict | None = None) -> list[Finding]
  detect_rules.references(articles: list[Article]) -> list[Reference]
  detect_similarity.detect(articles: list[Article], config: dict | None = None) -> list[Finding]
  detect_similarity.similar_pairs(articles: list[Article], min_score: float = 0.25) -> list[tuple[str, str, float]]
  detect_similarity.term_index(articles: list[Article]) -> dict
  ai_assist.verify_quotes(findings: list[Finding], articles: list[Article]) -> list[Finding]
  ai_assist.build_prompt(finding: Finding, articles: list[Article], kind: str = "unified_wording") -> str
  ai_assist.draft(finding: Finding, articles: list[Article], kind: str = "unified_wording") -> dict
  evaluate.evaluate(findings: list[Finding], labels_path: str) -> dict

Article (dict):
{
  "id": "CAR-OPS|4",              # "<regulation>|<article_no>", unique
  "regulation": "CAR-OPS",
  "article_no": "4",               # as printed; Arabic-Indic digits converted to 0-9; Arabic ordinal words kept (e.g. "الأولى")
  "title": "Retention of records",
  "text": "...",                   # body text, original wording (only NFKC-normalized). Quotes must be exact substrings of this.
  "lang": "en" | "ar",
  "source_file": "CAR-OPS.pdf",
  "page": 3 | null,
  "is_definitions": true | false,
  "extra": {}                      # optional. CAR-DEF terms: {"icao_source": "AN14/I", "kind": "glossary_term"}
}
CAR-DEF is a 3-column glossary table (term | definition | ICAO source), NOT "X means ..." text.
ingest.py turns it into ONE Article PER TERM: article_no = title = the term, text = the definition, extra.kind = "glossary_term".
  e.g. {"id":"CAR-DEF|Apron (APN)","regulation":"CAR-DEF","article_no":"Apron (APN)","title":"Apron (APN)","text":"A defined area, on a land aerodrome, ...","page":20,"is_definitions":true,"extra":{"icao_source":"AN2, AN4, ...","kind":"glossary_term"}}

Regulation names (ingest.py shortens GCAA file names):
  CAR-DEF            "CAR-DEF – DEFINITIONS – ISSUE 02.pdf"        CAR I, glossary, ~1,440 terms
  CAR-BR             "CAR-BR - ISSUE 02.pdf"                        CAR I, "Article 1 - Title" ... Article 20; definitions in Article 3 as "(12) ‘certificate’ means ..."
  CAR-ADR            "CAR ADR- AERODROMES - ISSUE 11.pdf"           CAR IX, chapters with sections "4.7 Safety Management System (SMS)"
  CAR-AGA-PART-ADR   "CAR-AGA - PART-ADR - ISSUE 02.pdf"            CAR IX, EASA style "ADR.OR.C.020 Title", guidance "AMC1 ADR.OR.C.010(j) ...", "GM1 ADR.OR.C.040 ..."
Display format for humans:
  "CAR-BR · Art. 3"   "CAR-ADR · §4.7"   "CAR-AGA-PART-ADR · ADR.OR.C.020"   "CAR-DEF · term “Apron (APN)”"   (+ " · p. 20" when page is known)

Real-text facts every module must handle:
  - PDF text has stray spaces inside words ("aerodr ome", "operato r") and line breaks in mid-sentence. Never assume clean words.
  - Running headers/footers and tables of contents are already removed by ingest.py.
  - Definitions use curly quotes: ‘apron’ means ... (also ’ “ ” and straight quotes).
  - CAR-DEF's own rule: when a specific CAR defines a term differently, the specific CAR's definition takes precedence.
    Lookup order: specific publication -> cover regulation -> CAR-BR -> CAR-DEF. So a divergent definition is not automatically wrong: it must be justified or harmonized.
  - CAR-ADR (Issue 11) and CAR-AGA-PART-ADR (Issue 02) are both published under CAR IX and repeat many requirements, sometimes with different modal verbs.
    Verified example: CAR-ADR "Initial notification of a vacancy shall, within 5 working days, ..." vs CAR-AGA-PART-ADR "(c) Initial notification of a vacancy should, within 5 working days, ...".

Finding (dict):
{
  "id": "<type>:<sorted article ids joined by +>",   # deterministic, e.g. "conflicting_values:CAR-AIR|4+CAR-OPS|4"
  "type": one of TYPES,
  "severity": "high" | "medium" | "low",
  "article_ids": ["CAR-OPS|4", "CAR-AIR|4"],
  "evidence": [{"article_id": "CAR-OPS|4", "quote": "<exact sentence copied from article text>"}],
  "summary": "one line a human understands",
  "recommendation": one of RECOMMENDATIONS,
  "rationale": "one or two sentences: why this recommendation",
  "detector": "rules" | "similarity",
  "confidence": 0.0-1.0,
  "quotes_verified": true | false,   # set by ai_assist.verify_quotes
  "extra": {}                        # anything detector-specific
}
TYPES: conflicting_values, duplicate_requirement, partial_overlap, duplicate_definition, divergent_definition, definition_missing_from_def, broken_reference, unverifiable_reference, icao_edition_issue
RECOMMENDATIONS: move_to_common, cross_reference, harmonize_wording, keep_with_justification, correct_reference
Severity order for sorting: high > medium > low.

Reference (dict): {"from": "CAR-OPS|6", "to_regulation": "CAR-OPS", "to_article": "14", "resolved_id": "CAR-OPS|14" | null, "quote": "<sentence>"}

Review (dict, in reviews.json keyed by finding id):
{"decision": "pending"|"accepted"|"modified"|"rejected", "final_recommendation": one of RECOMMENDATIONS, "note": "", "reviewer": "", "ai_draft": {} | null, "timestamp": "ISO-8601"}

Text normalization used for MATCHING only (never for quotes):
  NFKC; lowercase; Arabic-Indic digits ٠-٩ -> 0-9; remove tashkeel U+064B-U+0652, U+0670 and tatweel U+0640; أ إ آ ٱ -> ا ; ى -> ي ; ة -> ه
  Sentence split: first join PDF line breaks inside a paragraph (replace "\n" with " " unless the next line starts a list item such as "(a)", "(1)", "1.", "-"),
  then re.split(r'(?<=[.!?؟؛;])\s+|\n+', text)
  Term matching key: also remove ALL whitespace and punctuation (because of PDF word splits), e.g. "aircraft operato r" -> "aircraftoperator".

Rules for every module:
  - Never crash on missing optional files or missing modules: catch, print a warning, continue.
  - Evidence quotes are copied from Article.text, never paraphrased.
  - The code must run with: python <module>.py data/mock_articles.json (detectors) or data/mock_findings.json (ai_assist).

FIXTURE data/mock_articles.json (fictional, for testing only; already in the repo):
[
 {"id": "CAR-DEF|Operator", "regulation": "CAR-DEF", "article_no": "Operator", "title": "Operator", "text": "A person, organisation or enterprise engaged in or offering to engage in an aircraft operation.", "lang": "en", "source_file": "mock", "page": 74, "is_definitions": true, "extra": {"icao_source": "AN6/I", "kind": "glossary_term"}},
 {"id": "CAR-DEF|Occurrence", "regulation": "CAR-DEF", "article_no": "Occurrence", "title": "Occurrence", "text": "Any safety-related event which endangers or could endanger an aircraft, its occupants or any other person.", "lang": "en", "source_file": "mock", "page": 73, "is_definitions": true, "extra": {"icao_source": "AN13", "kind": "glossary_term"}},
 {"id": "CAR-OPS|2", "regulation": "CAR-OPS", "article_no": "2", "title": "Definitions", "text": "\"Operator\" means any person or company that operates an aircraft for commercial air transport.\n\"Dry lease\" means an agreement under which the aircraft is operated under the certificate of the lessee.", "lang": "en", "source_file": "mock", "page": null, "is_definitions": true, "extra": {}},
 {"id": "CAR-OPS|3", "regulation": "CAR-OPS", "article_no": "3", "title": "Occurrence reporting", "text": "The operator shall notify the Authority of any occurrence within 72 hours of the event becoming known.", "lang": "en", "source_file": "mock", "page": null, "is_definitions": false, "extra": {}},
 {"id": "CAR-OPS|4", "regulation": "CAR-OPS", "article_no": "4", "title": "Retention of records", "text": "The operator shall retain all aircraft maintenance records for a period of 2 years after the aircraft has been permanently withdrawn from service.", "lang": "en", "source_file": "mock", "page": null, "is_definitions": false, "extra": {}},
 {"id": "CAR-OPS|5", "regulation": "CAR-OPS", "article_no": "5", "title": "Safety management system", "text": "The operator shall establish, implement and maintain a safety management system appropriate to the size and complexity of its operations. The system shall include a safety policy, hazard identification, risk assessment, safety assurance and safety promotion.", "lang": "en", "source_file": "mock", "page": null, "is_definitions": false, "extra": {}},
 {"id": "CAR-OPS|6", "regulation": "CAR-OPS", "article_no": "6", "title": "Operations manual", "text": "The operator shall maintain an operations manual in accordance with CAR-OPS Article 14 and ICAO Annex 6, Part I, 10th Edition.", "lang": "en", "source_file": "mock", "page": null, "is_definitions": false, "extra": {}},
 {"id": "CAR-AIR|2", "regulation": "CAR-AIR", "article_no": "2", "title": "Definitions", "text": "\"Occurrence\" means any safety-related event which endangers or could endanger an aircraft, its occupants or any other person.\n\"Dry lease\" means an arrangement in which an aircraft is leased without crew and operated under the lessee's certificate.", "lang": "en", "source_file": "mock", "page": null, "is_definitions": true, "extra": {}},
 {"id": "CAR-AIR|3", "regulation": "CAR-AIR", "article_no": "3", "title": "Reporting", "text": "The organisation shall report to the Authority any occurrence affecting airworthiness within 24 hours.", "lang": "en", "source_file": "mock", "page": null, "is_definitions": false, "extra": {}},
 {"id": "CAR-AIR|4", "regulation": "CAR-AIR", "article_no": "4", "title": "Maintenance records", "text": "Maintenance records shall be kept for a minimum of 3 years after the aircraft has been permanently withdrawn from service.", "lang": "en", "source_file": "mock", "page": null, "is_definitions": false, "extra": {}},
 {"id": "CAR-AIR|5", "regulation": "CAR-AIR", "article_no": "5", "title": "Safety management", "text": "The organisation shall establish, implement and maintain a safety management system appropriate to the size and complexity of the organisation. The system shall include a safety policy, hazard identification, risk assessment, safety assurance and safety promotion.", "lang": "en", "source_file": "mock", "page": null, "is_definitions": false, "extra": {}},
 {"id": "CAR-AIR|6", "regulation": "CAR-AIR", "article_no": "6", "title": "Standards", "text": "Continuing airworthiness shall be managed in accordance with ICAO Annex 6, Part I, 12th Edition and CAR-OPS Article 9.", "lang": "en", "source_file": "mock", "page": null, "is_definitions": false, "extra": {}},
 {"id": "CAR-AOC|2", "regulation": "CAR-AOC", "article_no": "2", "title": "سجلات الصيانة", "text": "يلتزم المشغل بالاحتفاظ بسجلات الصيانة لمدة (5) سنوات على الأقل.", "lang": "ar", "source_file": "mock", "page": null, "is_definitions": false, "extra": {}}
]

EXPECTED FINDINGS on the mock (all modules together):
  conflicting_values       CAR-OPS|4 + CAR-AIR|4 + CAR-AOC|2   (maintenance records 2 / 3 / 5 years)
  conflicting_values       CAR-OPS|3 + CAR-AIR|3               (occurrence reporting 72 h vs 24 h)
  broken_reference         CAR-OPS|6 -> CAR-OPS Article 14 (does not exist)
  broken_reference         CAR-AIR|6 -> CAR-OPS Article 9  (does not exist)
  icao_edition_issue       CAR-OPS|6 + CAR-AIR|6               (Annex 6: 10th vs 12th edition)
  duplicate_requirement    CAR-OPS|5 + CAR-AIR|5               (SMS)
  partial_overlap          CAR-OPS|4 + CAR-AIR|4               (acceptable if scored as duplicate)
  divergent_definition     "Operator"   CAR-OPS|2 vs CAR-DEF|Operator
  duplicate_definition     "Occurrence" CAR-AIR|2 vs CAR-DEF|Occurrence
  definition_missing_from_def "Dry lease" CAR-OPS|2 + CAR-AIR|2

FIXTURE data/mock_findings.json (fictional, for UI/AI testing only; already in the repo):
[
 {"id": "conflicting_values:CAR-AIR|4+CAR-AOC|2+CAR-OPS|4", "type": "conflicting_values", "severity": "high", "article_ids": ["CAR-OPS|4", "CAR-AIR|4", "CAR-AOC|2"], "evidence": [{"article_id": "CAR-OPS|4", "quote": "The operator shall retain all aircraft maintenance records for a period of 2 years after the aircraft has been permanently withdrawn from service."}, {"article_id": "CAR-AIR|4", "quote": "Maintenance records shall be kept for a minimum of 3 years after the aircraft has been permanently withdrawn from service."}, {"article_id": "CAR-AOC|2", "quote": "يلتزم المشغل بالاحتفاظ بسجلات الصيانة لمدة (5) سنوات على الأقل."}], "summary": "Maintenance record retention: CAR-OPS Art.4 = 2 years; CAR-AIR Art.4 = 3 years; CAR-AOC Art.2 = 5 years", "recommendation": "move_to_common", "rationale": "Same obligation stated with three different periods in three regulations; the expert must choose one value and state it once.", "detector": "rules", "confidence": 0.9, "quotes_verified": true, "extra": {"topic": "retention", "object": "maintenance"}},
 {"id": "divergent_definition:CAR-DEF|Operator+CAR-OPS|2", "type": "divergent_definition", "severity": "high", "article_ids": ["CAR-OPS|2", "CAR-DEF|Operator"], "evidence": [{"article_id": "CAR-OPS|2", "quote": "\"Operator\" means any person or company that operates an aircraft for commercial air transport."}, {"article_id": "CAR-DEF|Operator", "quote": "A person, organisation or enterprise engaged in or offering to engage in an aircraft operation."}], "summary": "\"Operator\" is defined differently in CAR-OPS Art.2 and CAR-DEF (term \"Operator\")", "recommendation": "harmonize_wording", "rationale": "CAR-DEF is the reference for definitions; align the CAR-OPS definition or justify a sector-specific meaning.", "detector": "similarity", "confidence": 0.42, "quotes_verified": true, "extra": {"term": "Operator", "similarity": 0.42}},
 {"id": "broken_reference:CAR-OPS|6", "type": "broken_reference", "severity": "high", "article_ids": ["CAR-OPS|6"], "evidence": [{"article_id": "CAR-OPS|6", "quote": "The operator shall maintain an operations manual in accordance with CAR-OPS Article 14 and ICAO Annex 6, Part I, 10th Edition."}], "summary": "CAR-OPS Art.6 refers to CAR-OPS Article 14, which does not exist", "recommendation": "correct_reference", "rationale": "The referenced article is not in the regulation; it may have been repealed, moved or renumbered.", "detector": "rules", "confidence": 0.95, "quotes_verified": true, "extra": {"to_regulation": "CAR-OPS", "to_article": "14"}}
]
=== END CONTRACT ===
```
