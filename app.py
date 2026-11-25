"""RegLens expert review screen (Streamlit).

Run:  streamlit run app.py
Works with only app.py + data/mock_*.json present. Switches to the real data files
(data/articles.json, data/findings.json, ...) automatically when they exist.
Official regulation text is shown read-only and is never modified.
"""
from __future__ import annotations

import difflib
import glob
import html
import io
import itertools
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------- optional modules
try:
    import ai_assist  # noqa: F401
except Exception:  # noqa: BLE001
    ai_assist = None
try:
    import detect_rules  # noqa: F401
except Exception:  # noqa: BLE001
    detect_rules = None
try:
    import detect_similarity  # noqa: F401
except Exception:  # noqa: BLE001
    detect_similarity = None

DATA = "data"
RAW = os.path.join(DATA, "raw")
P = {
    "articles": os.path.join(DATA, "articles.json"),
    "findings": os.path.join(DATA, "findings.json"),
    "references": os.path.join(DATA, "references.json"),
    "similarity": os.path.join(DATA, "similarity.json"),
    "terms": os.path.join(DATA, "terms.json"),
    "coverage": os.path.join(DATA, "coverage.json"),
    "evaluation": os.path.join(DATA, "evaluation.json"),
    "reviews": os.path.join(DATA, "reviews.json"),
    "mock_articles": os.path.join(DATA, "mock_articles.json"),
    "mock_findings": os.path.join(DATA, "mock_findings.json"),
}

LABELS = {
    "types": {
        "conflicting_values": "Conflicting values",
        "duplicate_requirement": "Duplicate requirement",
        "partial_overlap": "Partial overlap",
        "duplicate_definition": "Duplicate definition",
        "divergent_definition": "Divergent definition",
        "definition_missing_from_def": "Definition missing from CAR-DEF",
        "broken_reference": "Broken reference",
        "unverifiable_reference": "Unverifiable reference",
        "icao_edition_issue": "ICAO edition issue",
    },
    "recommendations": {
        "move_to_common": "Move to common regulation",
        "cross_reference": "Cross-reference",
        "harmonize_wording": "Harmonize wording",
        "keep_with_justification": "Keep with justification",
        "correct_reference": "Correct reference",
    },
    "decisions": {"pending": "Pending", "accepted": "Accepted", "modified": "Modified", "rejected": "Rejected"},
}
RECS = list(LABELS["recommendations"])
DEF_TYPES = {"duplicate_definition", "divergent_definition", "definition_missing_from_def"}
SEV_RANK = {"high": 0, "medium": 1, "low": 2}
SEV_ICON = {"high": "🔴 High", "medium": "🟠 Medium", "low": "🟢 Low"}
PAGE_SIZE = 50
CONTEXT_CHARS = 600
LONG_TEXT = 2 * CONTEXT_CHARS + 400
MODAL_RE = re.compile(r"\b(shall|must|should|may)\b", re.IGNORECASE)

st.set_page_config(layout="wide", page_title="RegLens")
st.markdown(
    """<style>
    mark{background:#ffe08a;color:inherit;padding:0 2px;border-radius:2px}
    .rl-modal{background:#c62828;color:#fff;font-weight:700;padding:0 3px;border-radius:3px}
    .rl-text{border:1px solid rgba(128,128,128,.35);border-radius:6px;padding:10px 12px;
             max-height:340px;overflow:auto;white-space:pre-wrap;line-height:1.55;font-size:15px}
    .rl-badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:13px;margin-right:6px;
              border:1px solid rgba(128,128,128,.45)}
    .rl-del{text-decoration:line-through;color:#c62828}
    .rl-add{font-weight:700;color:#1b7a3d}
    </style>""",
    unsafe_allow_html=True,
)


def stretch(fn, *args, **kwargs):
    """Call a Streamlit element full-width on new and old Streamlit versions."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:  # noqa: BLE001  (older Streamlit: width must be an int)
        return fn(*args, use_container_width=True, **kwargs)


# ---------------------------------------------------------------- loading (cached on file mtime)
def _mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


@st.cache_resource(show_spinner=False, max_entries=16)
def _read_json(path, mtime):  # mtime is part of the cache key; result is read-only
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_json(path, default):
    m = _mtime(path)
    if m is None:
        return default
    try:
        return _read_json(path, m)
    except Exception as e:  # noqa: BLE001
        st.warning(f"Could not read {path}: {e}")
        return default


@st.cache_resource(show_spinner=False, max_entries=8)
def _index(articles_path, a_m, mock_path, m_m):
    by_id = {}
    for path in (mock_path, articles_path):
        if not path:
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                for a in json.load(fh):
                    if isinstance(a, dict) and a.get("id"):
                        by_id[a["id"]] = a
        except Exception:  # noqa: BLE001
            pass
    return by_id


def load_reviews():
    try:
        with open(P["reviews"], encoding="utf-8") as fh:
            r = json.load(fh)
            return r if isinstance(r, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def save_reviews(reviews):
    os.makedirs(DATA, exist_ok=True)
    tmp = P["reviews"] + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(reviews, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, P["reviews"])


def update_review(fid, **changes):
    all_r = load_reviews()
    cur = all_r.get(fid) or {"decision": "pending", "final_recommendation": None, "note": "", "reviewer": "",
                             "ai_draft": None, "timestamp": ""}
    cur.update(changes)
    all_r[fid] = cur
    save_reviews(all_r)


# ---------------------------------------------------------------- display helpers
def _split_id(aid):
    if aid and "|" in aid:
        return aid.split("|", 1)
    return aid or "", ""


def disp(aid, with_page=False):
    """Human label for an article id (contract v2 display format)."""
    if not aid:
        return ""
    a = BY_ID.get(aid) or {}
    reg = a.get("regulation") or _split_id(aid)[0]
    no = str(a.get("article_no") or _split_id(aid)[1] or "")
    kind = (a.get("extra") or {}).get("kind")
    if kind == "glossary_term":
        label = f"{reg} · term “{no}”"
    elif not no:
        label = reg
    elif re.match(r"[A-Za-z]", no):
        label = f"{reg} · {no}"
    elif re.fullmatch(r"\d+(?:\.\d+)+", no):
        label = f"{reg} · §{no}"
    else:
        label = f"{reg} · Art. {no}"
    if with_page and a.get("page"):
        label += f" · p. {a['page']}"
    return label


def disp_title(aid, with_page=False):
    a = BY_ID.get(aid) or {}
    t = a.get("title") or ""
    label = disp(aid, with_page)
    if t and (a.get("extra") or {}).get("kind") != "glossary_term" and t != a.get("article_no"):
        return f"{label}: {t}"
    return label


def reg_of(aid):
    a = BY_ID.get(aid) or {}
    return a.get("regulation") or _split_id(aid)[0]


def type_label(t):
    return LABELS["types"].get(t, t or "")


def rec_label(r):
    return LABELS["recommendations"].get(r, r or "")


def is_ar(text):
    letters = re.findall(r"[^\W\d_]", text or "")
    if not letters:
        return False
    ar = sum(1 for c in letters if "\u0600" <= c <= "\u06ff")
    return ar / len(letters) > 0.3


def finding_pairs(f):
    regs = sorted({reg_of(a) for a in f.get("article_ids", []) if a})
    if len(regs) == 1:
        return [f"{regs[0]} (within)"]
    return [f"{a} ↔ {b}" for a, b in itertools.combinations(regs, 2)]


def _spans(text, quotes):
    spans = []
    for q in quotes:
        q = (q or "").strip()
        if not q:
            continue
        start = 0
        while True:
            i = text.find(q, start)
            if i < 0:
                break
            spans.append((i, i + len(q)))
            start = i + len(q)
    spans.sort()
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _esc(s, modals):
    s = html.escape(s)
    if modals:
        s = MODAL_RE.sub(lambda m: f'<span class="rl-modal">{m.group(0)}</span>', s)
    return s


def highlight(text, quotes, modals=False, spans=None):
    """HTML-escape text and wrap every exact quote occurrence in <mark>."""
    text = text or ""
    spans = _spans(text, quotes) if spans is None else spans
    out, pos = [], 0
    for s, e in spans:
        out.append(_esc(text[pos:s], modals))
        out.append("<mark>" + _esc(text[s:e], modals) + "</mark>")
        pos = e
    out.append(_esc(text[pos:], modals))
    return "".join(out)


def _box(inner, rtl):
    attrs = 'dir="rtl" style="text-align:right"' if rtl else 'dir="ltr"'
    st.markdown(f'<div class="rl-text" {attrs}>{inner}</div>', unsafe_allow_html=True)


def text_box(text, lang, quotes=(), modals=False):
    """Read-only text with highlighted quotes. Long texts show only the context around the quote."""
    text = text or ""
    rtl = lang == "ar" or (lang is None and is_ar(text))
    spans = _spans(text, quotes)
    if len(text) <= LONG_TEXT:
        _box(highlight(text, quotes, modals, spans), rtl)
        return
    if spans:
        s0, e0 = spans[0][0], spans[-1][1]
        if e0 - s0 > 4 * CONTEXT_CHARS:  # quotes far apart: centre on the first one
            e0 = spans[0][1]
        a, b = max(0, s0 - CONTEXT_CHARS), min(len(text), e0 + CONTEXT_CHARS)
    else:
        a, b = 0, 2 * CONTEXT_CHARS
    part = text[a:b]
    sub = [(max(s, a) - a, min(e, b) - a) for s, e in spans if e > a and s < b]
    inner = ("… " if a > 0 else "") + highlight(part, (), modals, sub) + (" …" if b < len(text) else "")
    _box(inner, rtl)
    with st.expander(f"Full text ({len(text):,} characters)"):
        _box(highlight(text, quotes, modals, spans), rtl)


def word_diff(a, b):
    """Return (html_a, html_b): words only in a struck through, words only in b in bold."""
    wa, wb = (a or "").split(), (b or "").split()
    sm = difflib.SequenceMatcher(None, [w.lower() for w in wa], [w.lower() for w in wb], autojunk=False)
    out_a, out_b = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        ta, tb = html.escape(" ".join(wa[i1:i2])), html.escape(" ".join(wb[j1:j2]))
        if op == "equal":
            out_a.append(ta)
            out_b.append(tb)
        else:
            if ta:
                out_a.append(f'<span class="rl-del">{ta}</span>')
            if tb:
                out_b.append(f'<span class="rl-add">{tb}</span>')
    return " ".join(out_a), " ".join(out_b)


def run_cmd(args):
    try:
        r = subprocess.run([sys.executable, *args], capture_output=True, text=True, timeout=900)
        return r.returncode, (r.stdout or "") + (("\n" + r.stderr) if r.stderr else "")
    except Exception as e:  # noqa: BLE001
        return 1, f"{type(e).__name__}: {e}"


def show_run(label, args):
    with st.spinner(f"{label}…"):
        code, out = run_cmd(args)
    st.session_state["last_run"] = (label, code, out)
    st.cache_resource.clear()


def short_id(fid):
    return fid if len(fid) <= 48 else fid[:45] + "…"


# ---------------------------------------------------------------- data
demo = st.session_state.get("demo", False)

real_articles = load_json(P["articles"], None)
mock_articles = load_json(P["mock_articles"], [])
if demo or not real_articles:
    articles = mock_articles or []
    articles_src = P["mock_articles"]
else:
    articles = real_articles
    articles_src = P["articles"]

findings = load_json(P["findings"], None)
findings_src = P["findings"]
if findings is None or (not findings and articles_src == P["mock_articles"]):
    # No real findings yet (or detectors not merged): show the mock findings for the demo.
    findings = load_json(P["mock_findings"], [])
    findings_src = P["mock_findings"]
if not isinstance(findings, list):
    findings = []
findings = [f for f in findings if isinstance(f, dict) and f.get("id")]
F_BY_ID = {f["id"]: f for f in findings}

# Lookup covers both real and mock articles, so a mismatched pair of files still displays.
BY_ID = _index(P["articles"] if real_articles else None, _mtime(P["articles"]),
               P["mock_articles"], _mtime(P["mock_articles"]))

references = load_json(P["references"], [])
similarity = load_json(P["similarity"], [])
terms = load_json(P["terms"], {})
coverage = load_json(P["coverage"], None)
evaluation = load_json(P["evaluation"], None)
reviews = load_reviews()


def status_of(fid):
    return (reviews.get(fid) or {}).get("decision", "pending")


# ---------------------------------------------------------------- header
st.title("RegLens · Regulation consistency review")
st.caption("Supports the expert's decision. Official text is never modified.")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Data")
    for key in ["articles", "findings", "references", "similarity", "terms", "coverage", "evaluation"]:
        obj = load_json(P[key], None)
        if obj is None:
            st.markdown(f"⚪ `{key}.json` — missing")
        else:
            n = len(obj) if hasattr(obj, "__len__") else ""
            st.markdown(f"🟢 `{key}.json` — {n}")
    st.caption(f"Showing articles from `{articles_src}` and findings from `{findings_src}`.")

    st.divider()
    st.subheader("Regulations")
    ups = st.file_uploader("Upload regulations", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
    if ups and st.button("Save and analyse", type="primary"):
        os.makedirs(RAW, exist_ok=True)
        for u in ups:
            with open(os.path.join(RAW, os.path.basename(u.name)), "wb") as fh:
                fh.write(u.getbuffer())
        raw_files = [p for p in glob.glob(os.path.join(RAW, "*")) if not os.path.basename(p).startswith(".")]
        with st.spinner("Ingesting and analysing…"):
            code, out1 = run_cmd(["ingest.py", *raw_files])
            code2, out2 = run_cmd(["run_pipeline.py"])
        st.session_state["last_run"] = ("Ingest + analysis", code or code2, out1 + "\n" + out2)
        st.session_state["demo"] = False
        st.cache_resource.clear()
        st.rerun()

    if st.button("Re-run analysis"):
        show_run("Analysis", ["run_pipeline.py", "--mock"] if demo else ["run_pipeline.py"])
        st.rerun()
    new_demo = st.toggle("Use demo data", value=demo, help="Runs the pipeline on the fictional mock articles.")
    if new_demo != demo:
        st.session_state["demo"] = new_demo
        show_run("Analysis", ["run_pipeline.py", "--mock"] if new_demo else ["run_pipeline.py"])
        st.rerun()
    if "last_run" in st.session_state:
        label, code, out = st.session_state["last_run"]
        with st.expander(f"{label} output ({'ok' if code == 0 else 'check output'})"):
            st.code(out[-8000:] or "(no output)")

    st.divider()
    st.subheader("Filters")
    all_types = sorted({f.get("type", "") for f in findings})
    all_regs = sorted({reg_of(aid) for f in findings for aid in f.get("article_ids", [])})
    pair_counts = Counter(p for f in findings for p in finding_pairs(f))
    all_pairs = [p for p, _ in pair_counts.most_common()]
    f_types = st.multiselect("Type", all_types, format_func=type_label)
    f_sev = st.multiselect("Severity", ["high", "medium", "low"], default=["high", "medium"],
                           format_func=lambda s: SEV_ICON[s])
    f_regs = st.multiselect("Regulation", all_regs)
    f_pairs = st.multiselect("Regulation pair", all_pairs, format_func=lambda p: f"{p} ({pair_counts[p]})")
    f_status = st.multiselect("Review status", list(LABELS["decisions"]), format_func=LABELS["decisions"].get)
    f_text = st.text_input("Search summary and quotes")


def keep(f):
    if f_types and f.get("type") not in f_types:
        return False
    if f_sev and f.get("severity") not in f_sev:
        return False
    if f_regs and not any(reg_of(aid) in f_regs for aid in f.get("article_ids", [])):
        return False
    if f_pairs and not any(p in f_pairs for p in finding_pairs(f)):
        return False
    if f_status and status_of(f["id"]) not in f_status:
        return False
    if f_text:
        hay = (f.get("summary", "") + " " + " ".join(e.get("quote", "") for e in f.get("evidence", []))).lower()
        if f_text.lower() not in hay:
            return False
    return True


shown = sorted([f for f in findings if keep(f)],
               key=lambda f: (SEV_RANK.get(f.get("severity"), 9), f.get("type", ""),
                              -float(f.get("confidence") or 0), f["id"]))


# ---------------------------------------------------------------- overview data + export
def overview_matrix():
    cnt = Counter()
    for f in findings:
        for p in finding_pairs(f):
            cnt[(type_label(f.get("type")), p)] += 1
    if not cnt:
        return None
    df = pd.Series(cnt).unstack(fill_value=0)
    df = df[df.sum().sort_values(ascending=False).index]
    df.index.name = "Finding type"
    return df


def terms_df():
    if not isinstance(terms, dict) or not terms:
        return None
    rows = []
    for key, entries in terms.items():
        entries = entries or []
        ids = [e.get("article_id", "") for e in entries]
        name = next((e.get("term") for e in entries if e.get("term")), key)
        rows.append({
            "term": name,
            "defined in": ", ".join(disp(i) for i in ids),
            "count": len(ids),
            "in CAR-DEF": "yes" if any("DEF" in reg_of(i).upper() for i in ids) else "no",
        })
    return pd.DataFrame(rows).sort_values(["count", "term"], ascending=[False, True])


def findings_export_df():
    rows = []
    for f in findings:
        r = reviews.get(f["id"]) or {}
        draft = r.get("ai_draft") or {}
        ids = f.get("article_ids", [])
        row = {
            "id": f["id"], "severity": f.get("severity"), "type": type_label(f.get("type")),
            "articles": ", ".join(disp(a) for a in ids),
            "source files": ", ".join(dict.fromkeys(str((BY_ID.get(a) or {}).get("source_file") or "") for a in ids)),
            "pages": ", ".join(str((BY_ID.get(a) or {}).get("page") or "") for a in ids),
            "summary": f.get("summary", ""),
        }
        for i, ev in enumerate(f.get("evidence", []), 1):
            row[f"quote_{i}"] = f"[{disp(ev.get('article_id'))}] {ev.get('quote', '')}"
        row.update({
            "recommendation": rec_label(f.get("recommendation")), "rationale": f.get("rationale", ""),
            "quotes_verified": f.get("quotes_verified"),
            "ai_label": (f.get("extra") or {}).get("ai_label", ""),
            "decision": r.get("decision", "pending"),
            "final_recommendation": rec_label(r.get("final_recommendation", "")),
            "note": r.get("note", ""), "reviewer": r.get("reviewer", ""), "timestamp": r.get("timestamp", ""),
            "pinned": bool(r.get("pinned")),
            "ai_draft_en": draft.get("proposed_text_en", ""), "ai_draft_ar": draft.get("proposed_text_ar", ""),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def excel_bytes(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="Findings", index=False)
        ov = overview_matrix()
        if ov is not None:
            ov.to_excel(xw, sheet_name="Overview")
        tdf = terms_df()
        if tdf is not None and len(tdf):
            tdf.to_excel(xw, sheet_name="Terms", index=False)
        if evaluation:
            ev_rows = [{"metric": k, "value": json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v}
                       for k, v in evaluation.items()]
            pd.DataFrame(ev_rows).to_excel(xw, sheet_name="Evaluation", index=False)
    return buf.getvalue()


# ---------------------------------------------------------------- metrics + export row
def metrics_row():
    regs = {a.get("regulation") for a in articles if isinstance(a, dict)}
    sev_counts = Counter(f.get("severity") for f in findings)
    reviewed = sum(1 for f in findings if status_of(f["id"]) != "pending")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Regulations", len(regs))
    c2.metric("Articles", f"{len(articles):,}")
    c3.metric("Findings", f"{len(findings):,}")
    c3.caption(f"🔴 {sev_counts['high']} · 🟠 {sev_counts['medium']} · 🟢 {sev_counts['low']}")
    c4.metric("Reviewed", f"{(reviewed / len(findings) * 100) if findings else 0:.0f}%")


with st.sidebar:
    st.divider()
    st.subheader("Export")
    try:
        if st.button("Prepare Excel / CSV"):
            xdf = findings_export_df()
            st.session_state["export"] = (excel_bytes(xdf), xdf.to_csv(index=False).encode("utf-8-sig"),
                                          datetime.now().strftime("%Y%m%d-%H%M"))
        if "export" in st.session_state:
            xb, cb, stamp = st.session_state["export"]
            stretch(st.download_button, "⬇ Excel report", xb, file_name=f"reglens-report-{stamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            stretch(st.download_button, "⬇ CSV", cb, file_name=f"reglens-findings-{stamp}.csv", mime="text/csv")
            st.caption(f"Prepared {stamp}. Press again after new reviews.")
    except Exception as e:  # noqa: BLE001
        st.error(f"Export not available: {e}")

tab_o, tab_f, tab_demo, tab_i, tab_t, tab_d, tab_e = st.tabs(
    ["Overview", "Findings", "Demo", "Impact", "Terms", "Draft check", "Evaluation"])


# ---------------------------------------------------------------- finding detail
def render_ai_draft(d):
    st.warning("AI draft — not approved. Requires expert review.")
    mode = d.get("mode")
    if mode in ("manual", "error"):
        if mode == "error":
            st.caption(f"AI call failed: {d.get('error', '')}")
        else:
            st.caption("No API key set. Copy this prompt into any chat assistant:")
        st.code(d.get("prompt", ""), language="markdown")
        return
    if d.get("model"):
        st.caption(f"Model: {d['model']}")
    for key, lang, label in (("proposed_text_en", "en", "Proposed text (EN)"),
                             ("proposed_text_ar", "ar", "Proposed text (AR)"),
                             ("proposed_definition", None, "Proposed definition")):
        if d.get(key):
            st.markdown(f"**{label}**")
            text_box(d[key], lang)
    if d.get("placement"):
        st.markdown(f"**Placement:** {d['placement']}")
    if d.get("changes_to_official_text"):
        st.caption(f"Changes to official text: {d['changes_to_official_text']}")
    ch = d.get("changes") or []
    if ch:
        st.markdown("**Changes per article**")
        stretch(st.dataframe, pd.DataFrame([{"article": disp(c.get("article_id")), "action": c.get("action", ""),
                                             "new text": c.get("new_text", "")} for c in ch]), hide_index=True)
    qs = d.get("open_questions_for_expert") or d.get("questions_for_expert") or []
    if qs:
        st.markdown("**Open questions for the expert**")
        for q in qs:
            st.markdown(f"- {q}")
    qc = d.get("quotes_check")
    if qc is not None:
        if qc.get("all_verified"):
            st.success("✓ All AI-cited quotes found in the source text")
        else:
            st.error(f"⚠ Some AI-cited quotes were not found in the source: {qc.get('failed')}")


def render_definition_diff(f, quotes_by, ids):
    pair = [i for i in ids if quotes_by.get(i)][:2]
    if len(pair) < 2:
        return
    qa, qb = quotes_by[pair[0]][0], quotes_by[pair[1]][0]
    ha, hb = word_diff(qa, qb)
    st.markdown("**Definitions compared** (struck through: only on the left · bold: only on the right)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"*{html.escape(disp(pair[0]))}*", unsafe_allow_html=True)
        _box(ha, is_ar(qa))
    with c2:
        st.markdown(f"*{html.escape(disp(pair[1]))}*", unsafe_allow_html=True)
        _box(hb, is_ar(qb))


def render_finding(f, ctx):
    fid = f["id"]
    k = f"{ctx}-{fid}"
    extra = f.get("extra") or {}
    modals = extra.get("kind") == "obligation_strength"
    st.subheader(f.get("summary") or fid)
    badges = [
        f'<span class="rl-badge">{html.escape(type_label(f.get("type")))}</span>',
        f'<span class="rl-badge">{SEV_ICON.get(f.get("severity"), html.escape(str(f.get("severity", ""))))}</span>',
        f'<span class="rl-badge">confidence {float(f.get("confidence") or 0):.2f}</span>',
        f'<span class="rl-badge">detector: {html.escape(str(f.get("detector", "")))}</span>',
    ]
    if modals:
        badges.append(f'<span class="rl-badge">shall vs should: {html.escape(str(extra.get("modal_a", "")))} / '
                      f'{html.escape(str(extra.get("modal_b", "")))}</span>')
    if extra.get("ai_label"):
        badges.append(f'<span class="rl-badge">AI second opinion: {html.escape(str(extra["ai_label"]))}</span>')
    st.markdown(" ".join(badges), unsafe_allow_html=True)
    if extra.get("ai_reason"):
        st.caption(f"AI reason: {extra['ai_reason']}")
    if f.get("quotes_verified"):
        st.success("✓ Quotes verified in source text")
    else:
        st.warning("⚠ Quote not found in source; check manually")

    r = reviews.get(fid) or {}
    pinned = bool(r.get("pinned"))
    new_pin = st.toggle("Pin for demo", value=pinned, key=f"pin-{k}")
    if new_pin != pinned:
        update_review(fid, pinned=new_pin, pinned_at=datetime.now(timezone.utc).isoformat() if new_pin else None)
        st.rerun()

    ids = list(f.get("article_ids") or [])
    quotes_by = {}
    for ev in f.get("evidence", []):
        quotes_by.setdefault(ev.get("article_id"), []).append(ev.get("quote", ""))
    for aid in quotes_by:
        if aid not in ids:
            ids.append(aid)

    def article_col(aid):
        a = BY_ID.get(aid)
        if not a:
            st.markdown(f"**{html.escape(disp(aid))}**")
            st.caption("Article not found in the loaded articles.")
            for q in quotes_by.get(aid, []):
                text_box(q, None, [q], modals)
            return
        src = " · ".join(x for x in [str(a.get("source_file") or ""), f"p. {a['page']}" if a.get("page") else ""] if x)
        if (a.get("extra") or {}).get("kind") == "glossary_term":
            st.markdown(f"**{html.escape(a.get('regulation', ''))} · term “{html.escape(str(a.get('article_no')))}”**")
            st.markdown(f"**{html.escape(str(a.get('title') or a.get('article_no')))}**")
            text_box(a.get("text", ""), a.get("lang"), quotes_by.get(aid, []), modals)
            icao = (a.get("extra") or {}).get("icao_source")
            st.caption(" · ".join(x for x in [f"ICAO source: {icao}" if icao else "", src] if x))
            return
        st.markdown(f"**{html.escape(disp_title(aid))}**")
        if src:
            st.caption(src)
        text_box(a.get("text", ""), a.get("lang"), quotes_by.get(aid, []), modals)

    first, rest = ids[:3], ids[3:]
    if first:
        cols = st.columns(len(first))
        for col, aid in zip(cols, first):
            with col:
                article_col(aid)
    if rest:
        with st.expander(f"{len(rest)} more articles"):
            for aid in rest:
                article_col(aid)

    if f.get("type") in DEF_TYPES:
        render_definition_diff(f, quotes_by, ids)

    st.markdown(f"**Recommended action: {rec_label(f.get('recommendation'))}**")
    if f.get("rationale"):
        st.markdown(f.get("rationale"))
    if extra:
        with st.expander("Detector details"):
            st.json(extra)

    # ---- expert review
    st.markdown("---")
    st.markdown("#### Expert review")
    if r.get("decision") and r.get("decision") != "pending":
        st.info(f"Saved: **{LABELS['decisions'].get(r['decision'], r['decision'])}** · "
                f"{rec_label(r.get('final_recommendation'))} · {r.get('reviewer') or 'unknown reviewer'} · "
                f"{r.get('timestamp', '')}")
    dec_opts = ["accepted", "modified", "rejected"]
    with st.form(f"review-{k}"):
        cur = r.get("decision") if r.get("decision") in dec_opts else "accepted"
        decision = st.radio("Decision", dec_opts, index=dec_opts.index(cur),
                            format_func=LABELS["decisions"].get, horizontal=True)
        rec0 = r.get("final_recommendation") or f.get("recommendation")
        final = st.selectbox("Final recommendation", RECS, index=RECS.index(rec0) if rec0 in RECS else 0,
                             format_func=rec_label)
        note = st.text_area("Note", value=r.get("note", ""))
        reviewer = st.text_input("Reviewer", value=r.get("reviewer", "") or st.session_state.get("reviewer", ""))
        if st.form_submit_button("Save review", type="primary"):
            update_review(fid, decision=decision, final_recommendation=final, note=note, reviewer=reviewer,
                          timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            st.session_state["reviewer"] = reviewer
            st.rerun()

    # ---- AI draft
    st.markdown("#### Unified wording")
    has_ai = ai_assist is not None and hasattr(ai_assist, "draft")
    if st.button("Draft unified wording with AI", key=f"draft-{k}", disabled=not has_ai):
        arts = [BY_ID[i] for i in ids if i in BY_ID]
        with st.spinner("Drafting…"):
            try:
                d = ai_assist.draft(f, arts)
            except Exception as e:  # noqa: BLE001
                d = {"mode": "error", "error": str(e), "prompt": ""}
        cur_r = load_reviews().get(fid) or {}
        update_review(fid, ai_draft=d,
                      final_recommendation=cur_r.get("final_recommendation") or f.get("recommendation"))
        st.rerun()
    if not has_ai:
        st.caption("module not available yet (ai_assist.py)")
    if r.get("ai_draft"):
        render_ai_draft(r["ai_draft"])


def finding_card(f, ctx, i):
    """One-line card with an Open button. Returns True when clicked."""
    c1, c2 = st.columns([6, 1])
    with c1:
        st.markdown(f"**{SEV_ICON.get(f.get('severity'), '')} · {html.escape(type_label(f.get('type')))}** — "
                    f"{html.escape(f.get('summary') or f['id'])}  \n"
                    f"<small>{html.escape(', '.join(disp(a) for a in f.get('article_ids', [])))}</small>",
                    unsafe_allow_html=True)
    with c2:
        return st.button("Open", key=f"open-{ctx}-{i}-{f['id']}")


def card_list(items, ctx):
    key = f"sel-{ctx}"
    for i, f in enumerate(items):
        if finding_card(f, ctx, i):
            st.session_state[key] = f["id"]
    fid = st.session_state.get(key)
    if fid and fid in F_BY_ID:
        st.markdown("---")
        if st.button("Close", key=f"close-{ctx}"):
            st.session_state.pop(key, None)
            st.rerun()
        render_finding(F_BY_ID[fid], ctx)


# ---------------------------------------------------------------- tab: overview
with tab_o:
    try:
        metrics_row()
        cov = coverage if isinstance(coverage, list) else []
        cov = [c for c in cov if isinstance(c, dict) and c.get("articles_a")]
        if cov:
            best = max(cov, key=lambda c: float(c.get("share_of_a_with_a_match") or 0))
            n_a = int(best.get("articles_a") or 0)
            x = round(float(best.get("share_of_a_with_a_match") or 0) * n_a)
            st.metric(f"{best.get('reg_a')} sections with a near-copy in {best.get('reg_b')}", f"{x:,} of {n_a:,}",
                      f"{float(best.get('share_of_a_with_a_match') or 0):.0%}", delta_color="off")
            with st.expander("Coverage per regulation pair"):
                stretch(st.dataframe, pd.DataFrame(cov), hide_index=True)
        ov = overview_matrix()
        if ov is None:
            st.info("No findings yet. Run the analysis (sidebar) or switch on demo data.")
        else:
            st.markdown("**Findings by type and regulation pair**")
            try:
                styled = ov.style.background_gradient(cmap="Reds", axis=None)
                stretch(st.dataframe, styled)
            except Exception:  # noqa: BLE001  (matplotlib missing)
                stretch(st.dataframe, ov)
        top = sorted([f for f in findings if f.get("severity") == "high"],
                     key=lambda f: (-float(f.get("confidence") or 0), f["id"]))[:5]
        if top:
            st.markdown("**Top high-severity findings**")
            card_list(top, "ov")
    except Exception as e:  # noqa: BLE001
        st.error(str(e))


# ---------------------------------------------------------------- tab: findings
with tab_f:
    try:
        if not findings:
            st.info("No findings yet. Run the analysis (sidebar) or switch on demo data.")
        elif not shown:
            st.info("No findings match the filters.")
        else:
            n_pages = max(1, (len(shown) + PAGE_SIZE - 1) // PAGE_SIZE)
            st.session_state.setdefault("page", 1)
            if st.session_state["page"] > n_pages:  # filters changed: stay inside the range
                st.session_state["page"] = n_pages
            c1, c2 = st.columns([1, 4])
            with c1:
                page = st.number_input("Page", min_value=1, max_value=n_pages, step=1, key="page") \
                    if n_pages > 1 else 1
            start = (int(page) - 1) * PAGE_SIZE
            page_items = shown[start:start + PAGE_SIZE]
            with c2:
                st.caption(f"{len(shown):,} of {len(findings):,} findings match the filters · "
                           f"showing {start + 1}–{start + len(page_items)} (page {int(page)} of {n_pages}) · "
                           "click a row to open it")
            rows = [{
                "ID": short_id(f["id"]),
                "Severity": SEV_ICON.get(f.get("severity"), f.get("severity", "")),
                "Type": type_label(f.get("type")),
                "Articles": ", ".join(disp(a) for a in f.get("article_ids", [])),
                "Summary": f.get("summary", ""),
                "Recommendation": rec_label(f.get("recommendation")),
                "Review": LABELS["decisions"].get(status_of(f["id"]), status_of(f["id"])),
            } for f in page_items]
            df = pd.DataFrame(rows)
            sel_idx = None
            try:
                ev = stretch(st.dataframe, df, hide_index=True, on_select="rerun",
                             selection_mode="single-row", key=f"findings_table_{int(page)}")
                rows_sel = ev.selection.rows if ev is not None and hasattr(ev, "selection") else []
                if rows_sel:
                    sel_idx = rows_sel[0]
            except TypeError:
                stretch(st.dataframe, df, hide_index=True)
            if sel_idx is None:
                sel_idx = st.selectbox(
                    "Open finding", range(len(page_items)),
                    format_func=lambda i: f"{SEV_ICON.get(page_items[i].get('severity'), '')} · "
                                          f"{page_items[i].get('summary', page_items[i]['id'])}")
            st.markdown("---")
            render_finding(page_items[sel_idx], "fd")
    except Exception as e:  # noqa: BLE001
        st.error(str(e))


# ---------------------------------------------------------------- tab: demo
with tab_demo:
    try:
        pins = sorted([(r.get("pinned_at") or "", fid) for fid, r in reviews.items()
                       if isinstance(r, dict) and r.get("pinned") and fid in F_BY_ID])
        if not pins:
            st.info("No pinned findings yet. Open a finding and switch on “Pin for demo”.")
        else:
            st.caption(f"{len(pins)} pinned finding(s), in pin order.")
            card_list([F_BY_ID[fid] for _, fid in pins], "demo")
    except Exception as e:  # noqa: BLE001
        st.error(str(e))


# ---------------------------------------------------------------- tab: impact
with tab_i:
    try:
        if not articles:
            st.info("No articles loaded.")
        else:
            ids_sorted = sorted(a["id"] for a in articles if isinstance(a, dict) and a.get("id"))
            aid = st.selectbox("Article", ids_sorted, format_func=disp_title)
            inv = [f for f in findings if aid in f.get("article_ids", [])]
            refs_in = [r for r in (references or []) if isinstance(r, dict) and r.get("resolved_id") == aid]
            sims = []
            for p in similarity or []:
                try:
                    a_, b_, s_ = p[0], p[1], float(p[2])
                except Exception:  # noqa: BLE001
                    continue
                if s_ >= 0.4 and aid in (a_, b_):
                    sims.append((b_ if a_ == aid else a_, s_))
            sims.sort(key=lambda x: -x[1])
            n = len(inv) + len(refs_in) + len(sims)
            st.markdown(f"### Changing this article affects {n} place{'s' if n != 1 else ''}")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"**Findings involving it ({len(inv)})**")
                for f in inv[:100]:
                    others = [disp(x) for x in f.get("article_ids", []) if x != aid]
                    st.markdown(f"- {SEV_ICON.get(f.get('severity'), '')} {type_label(f.get('type'))}: "
                                f"{f.get('summary', '')}" + (f" — with {', '.join(others)}" if others else ""))
                if not inv:
                    st.caption("None")
            with c2:
                st.markdown(f"**Referenced from ({len(refs_in)})**")
                for r in refs_in[:100]:
                    st.markdown(f"- {disp_title(r.get('from'), True)}")
                if not refs_in:
                    st.caption("None" if references else "references.json not available")
            with c3:
                st.markdown(f"**Similar articles ({len(sims)})**")
                for oid, s in sims[:100]:
                    st.markdown(f"- {disp_title(oid, True)} · score {s:.2f}")
                if not sims:
                    st.caption("None" if similarity else "similarity.json not available")
            a = BY_ID.get(aid)
            if a:
                with st.expander("Article text (read-only)"):
                    text_box(a.get("text", ""), a.get("lang"))
    except Exception as e:  # noqa: BLE001
        st.error(str(e))


# ---------------------------------------------------------------- tab: terms
with tab_t:
    try:
        tdf = terms_df()
        if tdf is None:
            st.info("terms.json not available yet. Run the analysis.")
        else:
            st.caption("Highlighted: defined more than once, or not in CAR-DEF.")
            only_issues = st.toggle("Only highlighted terms", value=len(tdf) > 500)
            view = tdf[(tdf["count"] > 1) | (tdf["in CAR-DEF"] == "no")] if only_issues else tdf

            def _hl(row):
                bad = row["count"] > 1 or row["in CAR-DEF"] == "no"
                return ["background-color: rgba(255,200,0,.25)" if bad else ""] * len(row)

            if len(view) <= 2000:
                stretch(st.dataframe, view.style.apply(_hl, axis=1), hide_index=True)
            else:
                stretch(st.dataframe, view, hide_index=True)
    except Exception as e:  # noqa: BLE001
        st.error(str(e))


# ---------------------------------------------------------------- tab: draft check
with tab_d:
    try:
        st.markdown("Paste a new draft article to check it against the loaded regulations before it is issued.")
        avail = [m for m in (detect_rules, detect_similarity) if m is not None and hasattr(m, "detect")]
        if not avail:
            st.caption("module not available yet (detect_rules.py / detect_similarity.py)")
        reg_name = st.text_input("Regulation name", value="DRAFT")
        draft_text = st.text_area("Paste a new draft article", height=180)
        if st.button("Check draft", disabled=not avail or not draft_text.strip()):
            reg = (reg_name or "DRAFT").strip() or "DRAFT"
            did = f"{reg}|1" if reg != "DRAFT" else "DRAFT|1"
            d_art = {"id": did, "regulation": reg, "article_no": "1", "title": "Draft", "text": draft_text,
                     "lang": "ar" if is_ar(draft_text) else "en", "source_file": "draft", "page": None,
                     "is_definitions": bool(re.search(r"\bmeans\b|يقصد", draft_text)), "extra": {}}
            pool = [a for a in articles if a.get("id") != did] + [d_art]
            hits = []
            with st.spinner("Checking the draft against the loaded regulations…"):
                for m in avail:
                    try:
                        hits += [x for x in (m.detect(pool, None) or []) if did in x.get("article_ids", [])]
                    except Exception as e:  # noqa: BLE001
                        st.error(f"{m.__name__}: {e}")
            if not hits:
                st.success("No duplication, overlap or conflict found for this draft.")
            for h in hits:
                others = ", ".join(disp(x) for x in h.get("article_ids", []) if x != did)
                score = (h.get("extra") or {}).get("score") or h.get("confidence")
                verb = {"duplicate_requirement": "repeats", "partial_overlap": "overlaps with",
                        "conflicting_values": "conflicts with"}.get(h.get("type"), "relates to")
                tail = {"duplicate_requirement": " — consider a cross-reference.",
                        "partial_overlap": " — consider harmonizing or cross-referencing.",
                        "conflicting_values": " — align the value."}.get(h.get("type"), "")
                st.warning(f"**{type_label(h.get('type'))}:** This draft {verb} {others}"
                           + (f" (score {float(score):.2f})" if score else "") + tail
                           + f"\n\n{h.get('summary', '')}")
    except Exception as e:  # noqa: BLE001
        st.error(str(e))


# ---------------------------------------------------------------- tab: evaluation
with tab_e:
    try:
        if not evaluation:
            st.info("Run evaluate.py after adding data/expert_labels.csv")
        else:
            n, k = evaluation.get("labels", 0), evaluation.get("found", 0)
            c1, c2 = st.columns(2)
            c1.metric("Expert-labeled cases found", f"{k} of {n}",
                      f"recall {float(evaluation.get('recall') or 0):.0%}", delta_color="off")
            acc = evaluation.get("expert_acceptance_rate")
            c2.metric("Expert acceptance rate", "—" if acc is None else f"{float(acc):.0%}")
            pt = evaluation.get("per_type") or {}
            if pt:
                st.markdown("**Per type**")
                rows = []
                for t, v in pt.items():
                    row = {"type": type_label(t)}
                    row.update(v if isinstance(v, dict) else {"value": v})
                    rows.append(row)
                stretch(st.dataframe, pd.DataFrame(rows), hide_index=True)
            missed = evaluation.get("missed") or []
            if missed:
                st.markdown(f"**Missed ({len(missed)})**")
                stretch(st.dataframe, pd.DataFrame(missed), hide_index=True)
    except Exception as e:  # noqa: BLE001
        st.error(str(e))
