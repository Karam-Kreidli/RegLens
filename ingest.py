"""RegLens ingestion: regulation files (pdf / docx / txt / md) -> list[Article].

Usage:
    python ingest.py data/raw/*            -> writes data/articles.json
    python ingest.py FILE ... --out PATH
Env:
    REGLENS_HEADER_RE  optional single regex for article headers (group 1 = number, group 2 = title)
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys
import unicodedata
from collections import Counter, OrderedDict

AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

HEADER_PATTERNS = [
    # H1: "Article 5 - Title", "المادة (5) العنوان", "Section 2.1: Title"
    re.compile(
        r"^\s*(?:Article|ARTICLE|Art\.|Section|SECTION|المادة|مادة|البند)\s*[\(\[]?\s*"
        r"([0-9٠-٩]+[A-Za-z]?(?:[.\-][0-9٠-٩A-Z]+)*)\s*[\)\]]?\s*(?:[-–—:.]\s*)?(.*)$"
    ),
    # H2: "المادة الأولى", "المادة الحادية عشرة" (ordinal words kept as article_no)
    re.compile(r"^\s*(?:المادة|مادة)\s+([^\s:\-–—.]+(?:\s+عشرة?)?)\s*(?:[-–—:.]\s*)?(.*)$"),
    # H3: "ADR.OR.C.020 Continued validity", "AMC1 ADR.OR.C.010(j) Application", "GM1 ADR.OR.C.040 ...",
    #     "M.A.302 Maintenance programme", "4.7 Safety Management System (SMS)"
    re.compile(
        r"^\s*((?:(?:AMC|GM|CS)\d*\s+)?(?:[A-Z]{1,4}\.)+\d+(?:\.[A-Z0-9]+)*(?:\([a-z0-9]+\))*"
        r"|\d+(?:\.\d+)+)\s+(\S.{0,150})$"
    ),
]

DEF_RE = re.compile(r"(definition|interpretation|تعريف|تعاريف|التعاريف|معاني)", re.I)
ARABIC_LETTER = re.compile(r"[\u0600-\u06FF]")
LATIN_LETTER = re.compile(r"[A-Za-z]")
PRESENTATION_FORMS = re.compile(r"[\uFB50-\uFDFF\uFE70-\uFEFF]")
TOC_LINE = re.compile(r"(\.\s*){5,}|_{5,}|(\.\s){4,}")          # table-of-contents leaders
PAGE_LINE = re.compile(r"\bPage\s*\d+\s*(of|/)\s*\d+\b|^\s*\d+\s*$", re.I)  # "Page 13 of 31", bare page numbers


def _noise_key(line):
    return re.sub(r"[\d\s]+", "", line.lower())


def _header_patterns():
    custom = os.environ.get("REGLENS_HEADER_RE")
    if custom:
        try:
            return [re.compile(custom)]
        except re.error as e:
            print(f"[ingest] bad REGLENS_HEADER_RE ({e}); using defaults", file=sys.stderr)
    return HEADER_PATTERNS


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")


def _norm_light(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower().translate(AR_DIGITS)
    s = re.sub(r"[\u064B-\u0652\u0670\u0640]", "", s)
    return s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي").replace("ة", "ه")


# ---------------------------------------------------------------- readers
def _read_pdf(path):
    """Return list of (page_no, line)."""
    from pypdf import PdfReader

    pages = []
    reader = PdfReader(path)
    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as e:  # noqa: BLE001
            print(f"[ingest] {os.path.basename(path)} p{i}: extract failed ({e})", file=sys.stderr)
            raw = ""
        if PRESENTATION_FORMS.search(raw):
            print(f"[ingest] WARNING {os.path.basename(path)} p{i}: Arabic presentation forms found; "
                  f"NFKC applied. If text still looks reversed, paste it into a .txt file.", file=sys.stderr)
        pages.append((i, _clean(raw).split("\n")))
    # Running headers/footers: lines (digits ignored) that repeat on >= 30% of pages.
    freq = Counter()
    for _, lines in pages:
        freq.update({_noise_key(l) for l in lines if l.strip()})
    limit = max(3, int(0.3 * len(pages)))
    noise = {k for k, n in freq.items() if n >= limit and k}
    out = []
    for i, lines in pages:
        for line in lines:
            if _noise_key(line) in noise or PAGE_LINE.search(line):
                continue
            out.append((i, line))
    return out


def _read_docx(path):
    import docx

    d = docx.Document(path)
    lines = []
    # Body order: paragraphs and tables interleaved.
    body = d.element.body
    for child in body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            text = "".join(t.text or "" for t in child.iter() if t.tag.endswith("}t"))
            lines.append(text)
        elif tag == "tbl":
            for row in child.iter():
                if row.tag.endswith("}tr"):
                    cells = []
                    for cell in row.iterchildren():
                        if cell.tag.endswith("}tc"):
                            cells.append("".join(t.text or "" for t in cell.iter() if t.tag.endswith("}t")).strip())
                    lines.append(" | ".join(c for c in cells if c))
    return [(None, l) for l in _clean("\n".join(lines)).split("\n")]


def _read_txt(path):
    for enc in ("utf-8-sig", "cp1256", "latin-1"):
        try:
            with open(path, encoding=enc) as f:
                return [(None, l) for l in _clean(f.read()).split("\n")]
        except UnicodeDecodeError:
            continue
    return []


def _read_lines(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _read_pdf(path)
    if ext == ".docx":
        return _read_docx(path)
    if ext in (".txt", ".md", ".text", ""):
        return _read_txt(path)
    print(f"[ingest] skipping unsupported file {path}", file=sys.stderr)
    return []


# ---------------------------------------------------------------- helpers
def _regulation_overrides():
    p = os.path.join("data", "regulations.csv")
    m = {}
    if os.path.exists(p):
        with open(p, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("file") and row.get("regulation"):
                    m[row["file"].strip()] = row["regulation"].strip()
    return m


def _lang(text):
    ar = len(ARABIC_LETTER.findall(text))
    la = len(LATIN_LETTER.findall(text))
    return "ar" if ar and ar / max(ar + la, 1) > 0.3 else "en"


def _match_header(line, patterns):
    if len(line) > 220:
        return None
    for p in patterns:
        m = p.match(line)
        if m:
            num = m.group(1).strip().translate(AR_DIGITS)
            title = (m.group(2) if m.lastindex and m.lastindex >= 2 else "") or ""
            # A sentence that merely starts with "Article 5 of this ..." is not a header.
            t = title.strip()
            if t and (t[0].islower() or re.match(r"^(of|from|in|to|and|or|من|في|الى|إلى|و)\s", t)):
                continue
            if p is HEADER_PATTERNS[1] and len(line) > 80:
                continue
            # H3 must contain a digit
            if not re.search(r"\d", num) and p is HEADER_PATTERNS[2]:
                continue
            return num, title.strip()
    return None


def parse_def_table(path, reg, term_max_x=170, def_max_x=520):
    """CAR-DEF is a 3-column table (term | definition | ICAO source). One Article per term:
    article_no = title = term, text = definition, extra.icao_source = source column."""
    try:
        import pdfplumber
    except ImportError:
        print("[ingest] pdfplumber not installed; CAR-DEF table parsing skipped (pip install pdfplumber)", file=sys.stderr)
        return []
    rows, last = [], None
    with pdfplumber.open(path) as pdf:
        for pn, page in enumerate(pdf.pages, start=1):
            h = page.height
            words = [w for w in page.extract_words(use_text_flow=False, keep_blank_chars=False)
                     if 50 < h - w["bottom"] and w["top"] > 75]  # drop header/footer bands
            cols = {"t": [], "d": [], "s": []}
            for w in words:
                k = "t" if w["x0"] < term_max_x else ("d" if w["x0"] < def_max_x else "s")
                cols[k].append(w)

            def lines_of(ws):
                L = []
                for w in sorted(ws, key=lambda w: (round(w["top"]), w["x0"])):
                    if L and abs(L[-1]["top"] - w["top"]) < 2.5:
                        L[-1]["words"].append(w)
                    else:
                        L.append({"top": w["top"], "words": [w]})
                for l in L:
                    l["text"] = " ".join(x["text"] for x in sorted(l["words"], key=lambda x: x["x0"]))
                return L

            def groups(L, gap=15):
                G = []
                for l in L:
                    if G and l["top"] - G[-1]["bottom"] <= gap:
                        G[-1]["lines"].append(l["text"]); G[-1]["bottom"] = l["top"]
                    else:
                        G.append({"top": l["top"], "bottom": l["top"], "lines": [l["text"]]})
                for g in G:
                    g["c"] = (g["top"] + g["bottom"]) / 2
                return G

            T = [dict(g, d=[], s=[], page=pn) for g in groups(lines_of(cols["t"]))]
            for k in ("d", "s"):
                for g in groups(lines_of(cols[k])):
                    if not T:
                        if last is not None:
                            last[k] += g["lines"]
                        continue
                    if last is not None and g["bottom"] < T[0]["top"] - 10:
                        last[k] += g["lines"]  # row continued from previous page
                        continue
                    best = min(T, key=lambda b: abs(b["c"] - g["c"]))
                    best[k] += g["lines"]
            rows += T
            if T:
                last = T[-1]
    out, seen = [], Counter()
    lead_src = re.compile(r"^(?:[,\s]*(?:AN\s?\d+(?:/[IVX]+)?|ATM|PANS[-\w/]*)[,\s]*)+(?=[A-Z])")
    for r in rows:
        term = re.sub(r"\s+", " ", " ".join(r["lines"])).strip()
        definition = lead_src.sub("", re.sub(r"\s+", " ", " ".join(r["d"])).strip())
        if not definition or len(term) > 90 or len(term) < 2:
            continue
        seen[term] += 1
        no = term if seen[term] == 1 else f"{term}-{seen[term]}"
        out.append({
            "id": f"{reg}|{no}", "regulation": reg, "article_no": no, "title": term,
            "text": definition, "lang": _lang(definition), "source_file": os.path.basename(path),
            "page": r["page"], "is_definitions": True,
            "extra": {"icao_source": " ".join(r["s"]).strip(), "kind": "glossary_term"},
        })
    return out


def parse_file(path, regulation=None):
    reg = regulation or os.path.splitext(os.path.basename(path))[0].strip().upper()
    if path.lower().endswith(".pdf") and re.search(r"(^|[^A-Z])DEF([^A-Z]|$)", reg):
        terms = parse_def_table(path, reg)
        if len(terms) >= 20:
            print(f"[ingest] {reg}: parsed as definitions table ({len(terms)} terms)")
            return terms
    lines = _read_lines(path)
    patterns = _header_patterns()
    arts, cur = [], None
    for page, line in lines:
        if TOC_LINE.search(line):
            continue
        h = _match_header(line, patterns)
        if h:
            title, first = h[1], []
            if len(title.split()) >= 9 or re.search(r"\b(shall|must|should|may|يجب|يلتزم)\b", title):
                title, first = "", [h[1]]  # "4.7.1 The aerodrome operator shall ..." -> sentence is body text
            cur = {"article_no": h[0], "title": title, "lines": first, "page": page}
            arts.append(cur)
        elif cur is not None:
            cur["lines"].append(line)

    if not arts:  # no headers: paragraphs
        text = "\n".join(l for _, l in lines)
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        arts = [{"article_no": f"¶{i}", "title": "", "lines": [p], "page": None} for i, p in enumerate(paras, 1)]

    out, seen = [], Counter()
    for a in arts:
        body = "\n".join(l.rstrip() for l in a["lines"]).strip()
        body = re.sub(r"\n{3,}", "\n\n", body)
        if not body:
            continue  # table of contents line or empty heading
        no = a["article_no"]
        seen[no] += 1
        if seen[no] > 1:
            no = f"{no}-{seen[no]}"
        title = a["title"]
        out.append({
            "id": f"{reg}|{no}",
            "regulation": reg,
            "article_no": no,
            "title": title,
            "text": body,
            "lang": _lang(title + " " + body),
            "source_file": os.path.basename(path),
            "page": a["page"],
            "is_definitions": bool(DEF_RE.search(_norm_light(title + " " + body[:200]))) or "DEF" in reg.upper(),
        })
    return out


def short_name(filename):
    """'CAR-DEF – DEFINITIONS – ISSUE 02.pdf' -> 'CAR-DEF'; 'CAR-AGA - PART-ADR - ISSUE 02.pdf' -> 'CAR-AGA-PART-ADR';
    'CAR ADR- AERODROMES - ISSUE 11.pdf' -> 'CAR-ADR'. Returns (name, issue or None)."""
    stem = os.path.splitext(os.path.basename(filename))[0].strip().upper()
    issue = re.search(r"ISSUE\s*0*(\d+)", stem)
    parts = [x.strip() for x in re.split(r"\s*[–—-]\s+|\s+[–—-]\s*", stem) if x.strip()]
    if not parts or not parts[0].startswith("CAR"):
        return re.sub(r"\s+", "-", stem), (issue.group(1) if issue else None)
    name = parts[0] + "".join("-" + x for x in parts[1:] if x.startswith("PART"))
    return re.sub(r"\s+", "-", name), (issue.group(1) if issue else None)


def parse_documents(paths: list[str]) -> list[dict]:
    overrides = _regulation_overrides()
    articles, used = [], {}
    for p in paths:
        if os.path.isdir(p) or os.path.basename(p).startswith("."):
            continue
        reg = overrides.get(os.path.basename(p))
        if not reg:
            reg, issue = short_name(p)
            if reg in used and used[reg] != p:
                reg = f"{reg}-I{issue}" if issue else f"{reg}-2"
                print(f"[ingest] WARNING: two files map to the same regulation; using {reg} for {os.path.basename(p)}")
            used.setdefault(reg, p)
        try:
            articles.extend(parse_file(p, reg))
        except Exception as e:  # noqa: BLE001
            print(f"[ingest] ERROR reading {p}: {e}", file=sys.stderr)
    return articles


def main(argv):
    out = os.path.join("data", "articles.json")
    if "--out" in argv:
        i = argv.index("--out")
        out = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    paths = []
    for a in argv or ["data/raw/*"]:
        paths.extend(sorted(glob.glob(a)) or [a])
    arts = parse_documents(paths)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(arts, f, ensure_ascii=False, indent=1)

    per = OrderedDict()
    for a in arts:
        per[a["regulation"]] = per.get(a["regulation"], 0) + 1
    print(f"Wrote {len(arts)} articles to {out}")
    for reg, n in per.items():
        flag = "   <-- check header pattern" if n < 3 else ""
        print(f"  {reg:<20} {n:>5} articles{flag}")
    for a in arts[:3]:
        print(f"\n[{a['regulation']} · Art. {a['article_no']}] {a['title']} (p{a['page']}, {a['lang']}, defs={a['is_definitions']})")
        print("  " + a["text"][:200].replace("\n", " ") + ("…" if len(a["text"]) > 200 else ""))


if __name__ == "__main__":
    main(sys.argv[1:])
