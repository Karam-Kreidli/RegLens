"""RegLens expert review screen (Streamlit).

Run:  streamlit run app.py
Works with only app.py + data/mock_*.json present. Switches to the real data files
(data/articles.json, data/findings.json, ...) automatically when they exist.
Official regulation text is shown read-only and is never modified.
"""
from __future__ import annotations

import glob
import html
import io
import json
import os
import re
import subprocess
import sys
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
SEV_RANK = {"high": 0, "medium": 1, "low": 2}
SEV_ICON = {"high": "🔴 High", "medium": "🟠 Medium", "low": "🟢 Low"}

st.set_page_config(layout="wide", page_title="RegLens")
st.markdown(
    """<style>
    mark{background:#ffe08a;color:inherit;padding:0 2px;border-radius:2px}
    .rl-text{border:1px solid rgba(128,128,128,.35);border-radius:6px;padding:10px 12px;
             max-height:340px;overflow:auto;white-space:pre-wrap;line-height:1.55;font-size:15px}
    .rl-badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:13px;margin-right:6px;
              border:1px solid rgba(128,128,128,.45)}
    </style>""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- helpers
def _mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


@st.cache_data(show_spinner=False)
def _load_json_cached(path, mtime):  # mtime is the cache key
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_json(path, default):
    m = _mtime(path)
    if m is None:
        return default
    try:
        return _load_json_cached(path, m)
    except Exception as e:  # noqa: BLE001
        st.warning(f"Could not read {path}: {e}")
        return default


def load_reviews():
    try:
        with open(P["reviews"], encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return {}


def save_reviews(reviews):
    os.makedirs(DATA, exist_ok=True)
    tmp = P["reviews"] + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(reviews, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, P["reviews"])


def disp(aid, by_id=None):
    """'CAR-OPS|4' -> 'CAR-OPS · Art. 4'."""
    if not aid:
        return ""
    if "|" in aid:
        reg, no = aid.split("|", 1)
        return f"{reg} · Art. {no}"
    return aid


def disp_title(aid, by_id):
    a = by_id.get(aid)
    t = (a or {}).get("title") or ""
    return f"{disp(aid)}: {t}" if t else disp(aid)


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


def highlight(text, quotes):
    """HTML-escape text and wrap every exact quote occurrence in <mark>."""
    text = text or ""
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
    out, pos = [], 0
    for s, e in merged:
        out.append(html.escape(text[pos:s]))
        out.append("<mark>" + html.escape(text[s:e]) + "</mark>")
        pos = e
    out.append(html.escape(text[pos:]))
    return "".join(out)


def text_box(text, lang, quotes=()):
    rtl = lang == "ar" or (lang is None and is_ar(text))
    attrs = 'dir="rtl" style="text-align:right"' if rtl else 'dir="ltr"'
    st.markdown(f'<div class="rl-text" {attrs}>{highlight(text, quotes)}</div>', unsafe_allow_html=True)


def run_cmd(args):
    try:
        r = subprocess.run([sys.executable, *args], capture_output=True, text=True, timeout=600)
        return r.returncode, (r.stdout or "") + (("\n" + r.stderr) if r.stderr else "")
    except Exception as e:  # noqa: BLE001
        return 1, f"{type(e).__name__}: {e}"


def show_run(label, args):
    with st.spinner(f"{label}…"):
        code, out = run_cmd(args)
    st.session_state["last_run"] = (label, code, out)
    st.cache_data.clear()


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

# Lookup covers both real and mock articles, so a mismatched pair of files still displays.
by_id = {}
for a in (mock_articles or []) + (real_articles or []) + (articles or []):
    if isinstance(a, dict) and a.get("id"):
        by_id[a["id"]] = a

references = load_json(P["references"], [])
similarity = load_json(P["similarity"], [])
terms = load_json(P["terms"], {})
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
    for key in ["articles", "findings", "references", "similarity", "terms", "evaluation"]:
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
        code, out1 = run_cmd(["ingest.py", *raw_files])
        code2, out2 = run_cmd(["run_pipeline.py"])
        st.session_state["last_run"] = ("Ingest + analysis", code or code2, out1 + "\n" + out2)
        st.session_state["demo"] = False
        st.cache_data.clear()
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
    all_regs = sorted({aid.split("|")[0] for f in findings for aid in f.get("article_ids", [])})
    f_types = st.multiselect("Type", all_types, format_func=type_label)
    f_sev = st.multiselect("Severity", ["high", "medium", "low"], format_func=lambda s: SEV_ICON[s])
    f_regs = st.multiselect("Regulation", all_regs)
    f_status = st.multiselect("Review status", list(LABELS["decisions"]), format_func=LABELS["decisions"].get)
    f_text = st.text_input("Search summary and quotes")


def keep(f):
    if f_types and f.get("type") not in f_types:
        return False
    if f_sev and f.get("severity") not in f_sev:
        return False
    if f_regs and not any(aid.split("|")[0] in f_regs for aid in f.get("article_ids", [])):
        return False
    if f_status and status_of(f["id"]) not in f_status:
        return False
    if f_text:
        hay = (f.get("summary", "") + " " + " ".join(e.get("quote", "") for e in f.get("evidence", []))).lower()
        if f_text.lower() not in hay:
            return False
    return True


shown = sorted([f for f in findings if keep(f)],
               key=lambda f: (SEV_RANK.get(f.get("severity"), 9), f.get("type", ""), f["id"]))


# ---------------------------------------------------------------- export
def excel_bytes():
    rows = []
    for f in findings:
        r = reviews.get(f["id"]) or {}
        draft = r.get("ai_draft") or {}
        row = {
            "id": f["id"], "severity": f.get("severity"), "type": type_label(f.get("type")),
            "articles": ", ".join(disp(a) for a in f.get("article_ids", [])),
            "summary": f.get("summary", ""),
        }
        for i, ev in enumerate(f.get("evidence", []), 1):
            row[f"quote_{i}"] = f"[{disp(ev.get('article_id'))}] {ev.get('quote', '')}"
        row.update({
            "recommendation": rec_label(f.get("recommendation")), "rationale": f.get("rationale", ""),
            "quotes_verified": f.get("quotes_verified"),
            "decision": r.get("decision", "pending"),
            "final_recommendation": rec_label(r.get("final_recommendation", "")),
            "note": r.get("note", ""), "reviewer": r.get("reviewer", ""), "timestamp": r.get("timestamp", ""),
            "ai_draft_en": draft.get("proposed_text_en", ""), "ai_draft_ar": draft.get("proposed_text_ar", ""),
        })
        rows.append(row)
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="Findings", index=False)
        tdf = terms_df()
        if tdf is not None and len(tdf):
            tdf.to_excel(xw, sheet_name="Terms", index=False)
        if evaluation:
            ev_rows = [{"metric": k, "value": json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v}
                       for k, v in evaluation.items()]
            pd.DataFrame(ev_rows).to_excel(xw, sheet_name="Evaluation", index=False)
    return buf.getvalue(), df


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
            "in CAR-DEF": "yes" if any("DEF" in i.split("|")[0].upper() for i in ids) else "no",
        })
    return pd.DataFrame(rows).sort_values(["count", "term"], ascending=[False, True])


# ---------------------------------------------------------------- metrics + export row
try:
    regs = {a.get("regulation") for a in articles if isinstance(a, dict)}
    sev_counts = {s: sum(1 for f in findings if f.get("severity") == s) for s in ("high", "medium", "low")}
    reviewed = sum(1 for f in findings if status_of(f["id"]) != "pending")
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1.6, 1, 1.6])
    c1.metric("Regulations", len(regs))
    c2.metric("Articles", len(articles))
    c3.metric("Findings", len(findings),
              help="high / medium / low")
    c3.caption(f"🔴 {sev_counts['high']} · 🟠 {sev_counts['medium']} · 🟢 {sev_counts['low']}")
    c4.metric("Reviewed", f"{(reviewed / len(findings) * 100) if findings else 0:.0f}%")
    with c5:
        try:
            xbytes, xdf = excel_bytes()
            stamp = datetime.now().strftime("%Y%m%d-%H%M")
            st.download_button("⬇ Excel report", xbytes, file_name=f"reglens-report-{stamp}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
            st.download_button("⬇ CSV", xdf.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"reglens-findings-{stamp}.csv", mime="text/csv",
                               use_container_width=True)
        except Exception as e:  # noqa: BLE001
            st.error(f"Export not available: {e}")
except Exception as e:  # noqa: BLE001
    st.error(str(e))

tab_f, tab_i, tab_t, tab_d, tab_e = st.tabs(["Findings", "Impact", "Terms", "Draft check", "Evaluation"])


# ---------------------------------------------------------------- tab 1: findings
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
    if d.get("proposed_text_en"):
        st.markdown("**Proposed text (EN)**")
        text_box(d["proposed_text_en"], "en")
    if d.get("proposed_text_ar"):
        st.markdown("**Proposed text (AR)**")
        text_box(d["proposed_text_ar"], "ar")
    if d.get("placement"):
        st.markdown(f"**Placement:** {d['placement']}")
    ch = d.get("changes") or []
    if ch:
        st.markdown("**Changes per article**")
        st.dataframe(pd.DataFrame([{"article": disp(c.get("article_id")), "action": c.get("action", ""),
                                    "new text": c.get("new_text", "")} for c in ch]),
                     hide_index=True, use_container_width=True)
    qs = d.get("open_questions_for_expert") or []
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


def render_finding(f):
    fid = f["id"]
    st.subheader(f.get("summary") or fid)
    badges = [
        f'<span class="rl-badge">{html.escape(type_label(f.get("type")))}</span>',
        f'<span class="rl-badge">{SEV_ICON.get(f.get("severity"), f.get("severity", ""))}</span>',
        f'<span class="rl-badge">confidence {float(f.get("confidence") or 0):.2f}</span>',
        f'<span class="rl-badge">detector: {html.escape(str(f.get("detector", "")))}</span>',
    ]
    st.markdown(" ".join(badges), unsafe_allow_html=True)
    if f.get("quotes_verified"):
        st.success("✓ Quotes verified in source text")
    else:
        st.warning("⚠ Quote not found in source; check manually")

    ids = list(f.get("article_ids") or [])
    quotes_by = {}
    for ev in f.get("evidence", []):
        quotes_by.setdefault(ev.get("article_id"), []).append(ev.get("quote", ""))
    for aid in quotes_by:
        if aid not in ids:
            ids.append(aid)

    def article_col(aid):
        a = by_id.get(aid)
        st.markdown(f"**{disp_title(aid, by_id)}**")
        if not a:
            st.caption("Article not found in the loaded articles.")
            for q in quotes_by.get(aid, []):
                text_box(q, None, [q])
            return
        if a.get("page"):
            st.caption(f"{a.get('source_file', '')} · page {a['page']}")
        elif a.get("source_file"):
            st.caption(a["source_file"])
        text_box(a.get("text", ""), a.get("lang"), quotes_by.get(aid, []))

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

    st.markdown(f"**Recommended action: {rec_label(f.get('recommendation'))}**")
    if f.get("rationale"):
        st.markdown(f.get("rationale"))
    if f.get("extra"):
        with st.expander("Detector details"):
            st.json(f["extra"])

    # ---- expert review
    st.markdown("---")
    st.markdown("#### Expert review")
    r = reviews.get(fid) or {}
    if r.get("decision") and r.get("decision") != "pending":
        st.info(f"Saved: **{LABELS['decisions'].get(r['decision'], r['decision'])}** · "
                f"{rec_label(r.get('final_recommendation'))} · {r.get('reviewer') or 'unknown reviewer'} · "
                f"{r.get('timestamp', '')}")
    dec_opts = ["accepted", "modified", "rejected"]
    with st.form(f"review-{fid}"):
        cur = r.get("decision") if r.get("decision") in dec_opts else "accepted"
        decision = st.radio("Decision", dec_opts, index=dec_opts.index(cur),
                            format_func=LABELS["decisions"].get, horizontal=True)
        rec0 = r.get("final_recommendation") or f.get("recommendation")
        final = st.selectbox("Final recommendation", RECS, index=RECS.index(rec0) if rec0 in RECS else 0,
                             format_func=rec_label)
        note = st.text_area("Note", value=r.get("note", ""))
        reviewer = st.text_input("Reviewer", value=r.get("reviewer", "") or st.session_state.get("reviewer", ""))
        if st.form_submit_button("Save review", type="primary"):
            all_r = load_reviews()
            prev = all_r.get(fid) or {}
            all_r[fid] = {
                "decision": decision, "final_recommendation": final, "note": note, "reviewer": reviewer,
                "ai_draft": prev.get("ai_draft"),
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            save_reviews(all_r)
            st.session_state["reviewer"] = reviewer
            st.rerun()

    # ---- AI draft
    st.markdown("#### Unified wording")
    has_ai = ai_assist is not None and hasattr(ai_assist, "draft")
    if st.button("Draft unified wording with AI", key=f"draft-{fid}", disabled=not has_ai):
        arts = [by_id[i] for i in ids if i in by_id]
        with st.spinner("Drafting…"):
            try:
                d = ai_assist.draft(f, arts)
            except Exception as e:  # noqa: BLE001
                d = {"mode": "error", "error": str(e), "prompt": ""}
        all_r = load_reviews()
        cur_r = all_r.get(fid) or {"decision": "pending", "final_recommendation": f.get("recommendation"),
                                   "note": "", "reviewer": "", "timestamp": ""}
        cur_r["ai_draft"] = d
        all_r[fid] = cur_r
        save_reviews(all_r)
        st.rerun()
    if not has_ai:
        st.caption("module not available yet (ai_assist.py)")
    if r.get("ai_draft"):
        render_ai_draft(r["ai_draft"])


with tab_f:
    try:
        if not findings:
            st.info("No findings yet. Run the analysis (sidebar) or switch on demo data.")
        elif not shown:
            st.info("No findings match the filters.")
        else:
            rows = [{
                "ID": short_id(f["id"]),
                "Severity": SEV_ICON.get(f.get("severity"), f.get("severity", "")),
                "Type": type_label(f.get("type")),
                "Articles": ", ".join(disp(a) for a in f.get("article_ids", [])),
                "Summary": f.get("summary", ""),
                "Recommendation": rec_label(f.get("recommendation")),
                "Review": LABELS["decisions"].get(status_of(f["id"]), status_of(f["id"])),
            } for f in shown]
            df = pd.DataFrame(rows)
            st.caption(f"{len(shown)} of {len(findings)} findings · click a row to open it")
            sel_idx = None
            try:
                ev = st.dataframe(df, hide_index=True, use_container_width=True, on_select="rerun",
                                  selection_mode="single-row", key="findings_table")
                rows_sel = ev.selection.rows if ev is not None else []
                if rows_sel:
                    sel_idx = rows_sel[0]
            except TypeError:
                st.dataframe(df, hide_index=True, use_container_width=True)
            if sel_idx is None:
                pick = st.selectbox("Open finding", range(len(shown)),
                                    format_func=lambda i: f"{SEV_ICON.get(shown[i].get('severity'), '')} · "
                                                          f"{shown[i].get('summary', shown[i]['id'])}")
                sel_idx = pick
            st.markdown("---")
            render_finding(shown[sel_idx])
    except Exception as e:  # noqa: BLE001
        st.error(str(e))


# ---------------------------------------------------------------- tab 2: impact
with tab_i:
    try:
        if not articles:
            st.info("No articles loaded.")
        else:
            ids_sorted = sorted(by_id.keys() if not articles else [a["id"] for a in articles if a.get("id")])
            aid = st.selectbox("Article", ids_sorted, format_func=lambda i: disp_title(i, by_id))
            inv = [f for f in findings if aid in f.get("article_ids", [])]
            refs_in = [r for r in (references or []) if r.get("resolved_id") == aid]
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
                for f in inv:
                    others = [disp(x) for x in f.get("article_ids", []) if x != aid]
                    st.markdown(f"- {SEV_ICON.get(f.get('severity'), '')} {type_label(f.get('type'))}: "
                                f"{f.get('summary', '')}" + (f" — with {', '.join(others)}" if others else ""))
                if not inv:
                    st.caption("None")
            with c2:
                st.markdown(f"**Referenced from ({len(refs_in)})**")
                for r in refs_in:
                    st.markdown(f"- {disp_title(r.get('from'), by_id)}")
                if not refs_in:
                    st.caption("None" if references else "references.json not available")
            with c3:
                st.markdown(f"**Similar articles ({len(sims)})**")
                for oid, s in sims:
                    st.markdown(f"- {disp_title(oid, by_id)} · score {s:.2f}")
                if not sims:
                    st.caption("None" if similarity else "similarity.json not available")
            a = by_id.get(aid)
            if a:
                with st.expander("Article text (read-only)"):
                    text_box(a.get("text", ""), a.get("lang"))
    except Exception as e:  # noqa: BLE001
        st.error(str(e))


# ---------------------------------------------------------------- tab 3: terms
with tab_t:
    try:
        tdf = terms_df()
        if tdf is None:
            st.info("terms.json not available yet. Run the analysis.")
        else:
            st.caption("Highlighted: defined more than once, or not in CAR-DEF.")

            def _hl(row):
                bad = row["count"] > 1 or row["in CAR-DEF"] == "no"
                return ["background-color: rgba(255,200,0,.25)" if bad else ""] * len(row)

            st.dataframe(tdf.style.apply(_hl, axis=1), hide_index=True, use_container_width=True)
    except Exception as e:  # noqa: BLE001
        st.error(str(e))


# ---------------------------------------------------------------- tab 4: draft check
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
            is_def = bool(re.search(r"\bmeans\b|يقصد|:", draft_text))
            d_art = {"id": did, "regulation": reg, "article_no": "1", "title": "Draft", "text": draft_text,
                     "lang": "ar" if is_ar(draft_text) else "en", "source_file": "draft", "page": None,
                     "is_definitions": is_def and bool(re.search(r"\bmeans\b|يقصد", draft_text))}
            pool = [a for a in articles if a.get("id") != did] + [d_art]
            hits = []
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


# ---------------------------------------------------------------- tab 5: evaluation
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
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            missed = evaluation.get("missed") or []
            if missed:
                st.markdown(f"**Missed ({len(missed)})**")
                st.dataframe(pd.DataFrame(missed), hide_index=True, use_container_width=True)
    except Exception as e:  # noqa: BLE001
        st.error(str(e))
