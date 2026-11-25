"""P3: definition consistency and cross-regulation requirement similarity.

Usage:
    python detect_similarity.py data/mock_articles.json
    python detect_similarity.py data/mock_articles.json --test
    python detect_similarity.py data/articles.json --config config.json
    python detect_similarity.py data/articles.json --stats

Config keys: dup (default 0.85), overlap (0.60), max_findings_per_type (300),
coverage_path (data/coverage.json; None disables writing), def_regulation
(auto: CAR-DEF or the first regulation containing DEF). Scores are review candidates,
not a legal determination. Matching normalization never changes source quotes.
Only scikit-learn and the standard library are used; term_index works without
scikit-learn. Article comparisons use sparse products in blocks of 128 rows.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
import heapq
from itertools import combinations
import json
import math
from pathlib import Path
import re
import sys
import unicodedata

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = cosine_similarity = None

Article = dict
Finding = dict
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩أإآٱىة", "0123456789اااايه")
_MARKS = re.compile(r"[\u064b-\u0652\u0670\u0640]")
_SENTENCES = re.compile(
    r"(?<=[.!?؟؛;])\s+|\n[ \t]*\n+|\n(?=[ \t]*(?:\([a-z0-9]+\)|[0-9]+[.)]|[-–•]))",
    re.I,
)
_QUOTED = re.compile(
    r'''["“‘'«]([^"”’'»\n]{2,80})["”’'»]\s*'''
    r"(?:means|shall mean|refers to|is defined as)\s+", re.I,
)
_UNQUOTED = re.compile(
    r"^\s*(?:\(?[a-z0-9]{1,3}\)\s*)?"
    r"([A-Z][A-Za-z\- ]{2,50}?)\s+means\s+(.+)$",
)
_ARABIC = re.compile(
    r"^\s*(?:[-–•]|\(?[0-9٠-٩a-zأ-ي]{1,3}[\).])?\s*"
    r"([^:\n]{2,40}?)\s*:\s+(.{10,})$",
)
_ABBREVIATION = re.compile(r"\s*\(([A-Z][A-Z0-9 /-]{1,14})\)\s*$")
_MODAL = re.compile(
    r"(?<!\w)(?P<prefix>لا\s+)?(?P<modal>is required to|is recommended(?: to)?|"
    r"shall|must|should|may|يجب|يجوز|ينبغي)(?P<negative>\s+not)?(?!\w)", re.I,
)
_STRONG = {"shall", "must", "is required to", "يجب"}
_RANK = {"high": 0, "medium": 1, "low": 2}


def _warn(message: str) -> None:
    print(f"[similarity] WARNING: {message}", file=sys.stderr)


def _norm(text: str) -> str:
    return _MARKS.sub("", unicodedata.normalize("NFKC", text).lower().translate(_DIGITS))


def _term_key(term: str) -> str:
    term = _ABBREVIATION.sub("", term)
    acronym = term.strip('"“”‘’\'«» ').isupper()
    words = re.findall(r"[^\W\d_]+", _norm(term))
    for i, word in enumerate(words):
        if word.startswith("ال") and len(word) >= 5:
            words[i] = word[2:]
        elif not acronym and len(word) >= 5 and word.isascii() and word.endswith("s"):
            words[i] = word[:-1]
    return "".join(words)


def _aliases(term: str) -> tuple[str, ...]:
    match = _ABBREVIATION.search(term)
    return (_term_key(match.group(1)),) if match else ()


@dataclass(frozen=True)
class _Definition:
    article: Article
    term: str
    key: str
    body: str
    quote: str
    aliases: tuple[str, ...] = ()
    see_also: str | None = None


def _articles(articles: list[Article]) -> list[Article]:
    # Ignore empty article bodies without mutating the caller's dictionaries.
    return sorted(
        (a for a in articles if isinstance(a, dict) and a.get("id") and a.get("regulation")
         and isinstance(a.get("text"), str) and a["text"].strip()),
        key=lambda a: a["id"],
    )


def _definitions(articles: list[Article]) -> list[_Definition]:
    definitions = []
    seen = set()
    for article in _articles(articles):
        text = article["text"]
        if (article.get("extra") or {}).get("kind") == "glossary_term":
            term = article.get("title") or article.get("article_no", "")
            reference = re.match(r"\s*See\s*:\s*(.+)", text, re.I | re.S)
            definitions.append(_Definition(article, term, _term_key(term), text.strip(), text.strip(),
                                           _aliases(term), reference.group(1).strip() if reference else None))
            continue
        # Record offsets in original text: PDF wraps remain in evidence spans.
        markers = []
        for match in _QUOTED.finditer(text):
            line_start = text.rfind("\n", 0, match.start()) + 1
            start = line_start if not markers or markers[-1][2] <= line_start else match.start()
            numbered = re.search(r"\(\d+\)\s*$", text[:match.start()])
            if numbered:
                start = numbered.start()
            markers.append((start, match.group(1).strip(), match.end(), True))
        offset = 0
        for line in text.splitlines(keepends=True):
            match = _UNQUOTED.fullmatch(line.rstrip("\r\n"))
            if match is None and _QUOTED.search(line) is None:
                arabic = _ARABIC.fullmatch(line.rstrip("\r\n"))
                if arabic and re.search(r"[\u0600-\u06ff]", arabic.group(1)):
                    match = arabic
            if match:
                markers.append((offset, match.group(1).strip(), offset + match.start(2),
                                bool(article.get("is_definitions", False))))
            offset += len(line)
        markers.sort()
        for i, (start, term, body_start, allowed) in enumerate(markers):
            if not allowed:
                continue
            end = markers[i + 1][0] if i + 1 < len(markers) else len(text)
            quote, body = text[start:end].strip(), text[body_start:end].strip()
            key = _term_key(term)
            identity = (article["id"], key, quote)
            if key and body and identity not in seen:
                seen.add(identity)
                definitions.append(_Definition(article, term, key, body, quote, _aliases(term)))
    return definitions


def term_index(articles: list[Article]) -> dict:
    """Group definitions and abbreviation aliases; text is always an original span."""
    result = defaultdict(list)
    for definition in _definitions(articles):
        entry = {
            "article_id": definition.article["id"],
            "term": definition.term,
            "text": definition.quote,
        }
        if definition.see_also:
            entry["see_also"] = definition.see_also
        result[definition.key].append(entry)
        for alias in definition.aliases:
            if alias and alias != definition.key:
                result[alias].append({**entry, "alias_of": definition.key})
    return dict(sorted(result.items()))


def _available() -> bool:
    if TfidfVectorizer is None or cosine_similarity is None:
        _warn("scikit-learn is unavailable; install requirements.txt to enable similarity scoring.")
        return False
    return True


def _vectors(texts: list[str], *, compact: bool = False):
    if not texts:
        return None, None
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1, max_features=300000,
    )
    try:
        normalized = [re.sub(r"\s+", "" if compact else " ", _norm(text)).strip() for text in texts]
        return vectorizer, vectorizer.fit_transform(normalized)
    except ValueError as exc:
        # Blank/very short normalized text can have no character n-grams.
        _warn(f"cannot build a TF-IDF vocabulary ({exc}); skipped these comparisons.")
        return None, None


def _score(value) -> float:
    value = float(value)
    # Sparse products can place identical texts a few ulps below one.
    return 1.0 if value >= 1.0 - 1e-12 else max(0.0, value)


def _pair_scores(articles: list[Article], min_score: float, *, top_k: int | None = None):
    """Score sparse blocks per regulation pair; optionally retain directional top-k matches."""
    if len(articles) < 2:
        return
    _, matrix = _vectors([a["text"] for a in articles])
    if matrix is None:
        return
    groups = defaultdict(list)
    for i, article in enumerate(articles):
        groups[article["regulation"]].append(i)

    def directional(left, right):
        transpose = matrix[right].T
        for start in range(0, len(left), 128):
            indices = left[start:start + 128]
            product = (matrix[indices] @ transpose).tocsr()
            product.sort_indices()
            if top_k is None:
                entries = product.tocoo()
                keep = entries.data >= min_score - 1e-12
                for row, col, value in zip(entries.row[keep], entries.col[keep], entries.data[keep]):
                    score = _score(value)
                    if score >= min_score:
                        yield indices[int(row)], right[int(col)], score
            else:
                for row, index in enumerate(indices):
                    a, b = product.indptr[row:row + 2]
                    values, columns = product.data[a:b], product.indices[a:b]
                    # Stable sorting resolves tied scores by article ID, not sparse insertion order.
                    for position in (-values).argsort(kind="stable")[:top_k]:
                        score = _score(values[position])
                        if score >= min_score:
                            yield index, right[int(columns[position])], score

    for reg_a, reg_b in combinations(sorted(groups), 2):
        left, right = groups[reg_a], groups[reg_b]
        if top_k is None:
            for i, j, score in directional(left, right):
                yield min(i, j), max(i, j), score
        else:
            best = {}
            for a, b in ((left, right), (right, left)):
                for i, j, score in directional(a, b):
                    best[min(i, j), max(i, j)] = score
            for (i, j), score in sorted(best.items()):
                yield i, j, score


def similar_pairs(articles: list[Article], min_score: float = 0.25) -> list[tuple[str, str, float]]:
    """Return up to 2,000 scored cross-regulation links, including definitions.

    min_score must be in (0, 1]. Unlike detect's requirement candidates, this
    impact index includes short articles and definitions.
    """
    if not math.isfinite(min_score) or not 0 < min_score <= 1:
        raise ValueError("min_score must be greater than zero and at most one")
    source = _articles(articles)
    if len(source) < 2 or not _available():
        return []
    # Keep bounded output memory even when a corpus contains many repeated texts.
    pairs = ((source[i]["id"], source[j]["id"], score)
             for i, j, score in _pair_scores(source, min_score))
    return heapq.nsmallest(2000, pairs, key=lambda pair: (-pair[2], pair[0], pair[1]))


def _label(article: Article) -> str:
    reg = article["regulation"]
    number = str(article.get("article_no", article["id"].split("|")[-1]))
    if (article.get("extra") or {}).get("kind") == "glossary_term":
        return f'{reg} term "{article.get("title") or number}"'
    if re.match(r"[A-Z]", number):
        return f"{reg} {number}"
    if re.fullmatch(r"\d+(?:\.\d+)+", number):
        return f"{reg} §{number}"
    return f"{reg} Art.{number}"


def _finding(kind, articles, evidence, summary, recommendation, rationale, confidence, extra, severity="medium"):
    ids = sorted({article["id"] for article in articles})
    return {
        "id": f"{kind}:{'+'.join(ids)}",
        "type": kind,
        "severity": severity,
        "article_ids": ids,
        "evidence": evidence,
        "summary": summary,
        "recommendation": recommendation,
        "rationale": rationale,
        "detector": "similarity",
        "confidence": _score(confidence),
        "quotes_verified": False,  # The pipeline/ai_assist owns verification.
        "extra": extra,
    }


def _definition_findings(articles, definitions, config):
    all_definitions = definitions
    definitions = [definition for definition in definitions if not definition.see_also]
    if not definitions:
        return []
    regulations = sorted({a["regulation"] for a in articles})
    requested = config.get("def_regulation")
    if requested:
        baseline = next((reg for reg in regulations if reg.casefold() == str(requested).casefold()), None)
    else:
        baseline = next((reg for reg in regulations if reg.upper() == "CAR-DEF"), None)
        baseline = baseline or next((reg for reg in regulations if "DEF" in reg.upper()), None)
    if baseline is None:
        _warn("definitions regulation not loaded; CAR-DEF comparisons skipped (term index is still available).")
        return []
    linked = {key for definition in all_definitions
              if definition.see_also and definition.article["regulation"] == baseline
              for key in {definition.key, *definition.aliases}}
    _, matrix = _vectors([definition.body for definition in definitions], compact=True)
    if matrix is None:
        return []
    common, local, canonical = defaultdict(list), defaultdict(list), defaultdict(list)
    for i, definition in enumerate(definitions):
        if definition.article["regulation"] == baseline:
            canonical[definition.key].append(i)
            for key in {definition.key, *definition.aliases}:
                common[key].append(i)
        else:
            local[definition.key].append(i)
    findings = []
    for key, indices in sorted(canonical.items()):
        for i, j in combinations(indices, 2):
            left, right = definitions[i], definitions[j]
            if left.article["id"] == right.article["id"] or re.sub(r"\W", "", _norm(left.body)) == re.sub(r"\W", "", _norm(right.body)):
                continue
            score = _score(cosine_similarity(matrix[i:i + 1], matrix[j:j + 1])[0, 0])
            findings.append(_finding(
                "divergent_definition", [left.article, right.article],
                [{"article_id": d.article["id"], "quote": d.quote} for d in (left, right)],
                f'"{left.term}" has two definitions within {baseline}', "harmonize_wording",
                f"{baseline} itself holds two definitions; state the domain of each or merge.", score,
                {"term": left.term, "similarity": score, "alias_used": False},
            ))
    for key, indices in sorted(local.items()):
        reference_indices = sorted({j for i in indices for match_key in {key, *definitions[i].aliases}
                                    for j in common.get(match_key, [])})
        if not reference_indices:
            if any(match_key in linked for i in indices for match_key in {key, *definitions[i].aliases}):
                continue
            missing = [definitions[i] for i in indices]
            if len({definition.article["regulation"] for definition in missing}) >= 2:
                term = missing[0].term
                findings.append(_finding(
                    "definition_missing_from_def", [d.article for d in missing],
                    [{"article_id": d.article["id"], "quote": d.quote} for d in missing],
                    f'"{term}" is defined in multiple regulations but absent from {baseline}',
                    "move_to_common", f"Add one definition to {baseline} and reference it from each regulation.",
                    0.7, {"term": term, "similarity": None, "alias_used": False},
                ))
            continue
        for i in indices:
            definition = definitions[i]
            # Different local entries can carry different aliases; match each independently.
            references = sorted({j for match_key in {key, *definition.aliases} for j in common.get(match_key, [])})
            if not references:
                continue
            scores = cosine_similarity(matrix[i:i + 1], matrix[references]).ravel()
            best = int(scores.argmax())
            reference = definitions[references[best]]
            score = _score(scores[best])
            duplicate = score >= 0.80
            divergent = score < 0.50
            summary = (
                f'"{definition.term}" in {_label(definition.article)} repeats {_label(reference.article)}'
                if duplicate else
                f'"{definition.term}" is defined differently in {_label(definition.article)} and {_label(reference.article)}'
            )
            findings.append(_finding(
                "duplicate_definition" if duplicate else "divergent_definition",
                [definition.article, reference.article],
                [{"article_id": d.article["id"], "quote": d.quote} for d in (definition, reference)],
                summary, "cross_reference" if duplicate else "keep_with_justification" if divergent else "harmonize_wording",
                f"Delete the local definition and refer to {baseline}." if duplicate else
                f"Different meaning. {baseline} allows a specific CAR's definition to take precedence; "
                "state the domain and confirm the difference is intentional." if divergent else
                f"Same concept, different wording; align with {baseline} or document the sector-specific scope.",
                score, {"term": definition.term, "similarity": score, "alias_used": definition.key != reference.key},
                "high" if divergent else "medium",
            ))
    return findings


def _containment(left: str, right: str) -> float:
    words = [re.findall(r"[^\W_]+", _norm(text)) for text in (left, right)]
    shorter, longer = sorted(words, key=len)
    unique = set(shorter)
    return len(unique.intersection(longer)) / len(unique) if unique else 0.0


def _sentences(text: str) -> list[str]:
    boundaries = []
    for match in _SENTENCES.finditer(text):
        # Do not split the article number off a reference such as "Art. 4".
        if re.search(r"\b(?:Art|No|Sec|Dr|Mr)\.$", text[:match.start()], re.I):
            continue
        boundaries.append(match)
    result, start = [], 0
    for boundary in boundaries:
        if text[start:boundary.start()].strip():
            result.append(text[start:boundary.start()].strip())
        start = boundary.end()
    if text[start:].strip():
        result.append(text[start:].strip())
    substantive = [sentence for sentence in result
                   if len(re.findall(r"[^\W_]+", sentence)) >= 5 and not sentence.isupper()]
    # Exact headings/conjunctions otherwise win cosine=1 over the real obligation.
    return substantive or [max(result, key=len)]


def _modal_view(text):
    modals = []

    def replace(match):
        modal = match.group("modal").lower()
        modals.append({"modal": modal, "strong": modal in _STRONG,
                       "negative": bool(match.group("negative") or match.group("prefix"))})
        return " zzmodalmarkerzz "

    normalized = re.sub(r"\s+", " ", _norm(text))
    normalized = re.sub(r"^\s*(?:\([a-z0-9]+\)|[0-9]+[.)]|[-–•])\s*", "", normalized)
    tokens = re.findall(r"[^\W_]+", _MODAL.sub(replace, normalized))
    positions = [i for i, token in enumerate(tokens) if token == "zzmodalmarkerzz"]
    return tokens, dict(zip(positions, modals))


def _obligation_change(left, right):
    a, modals_a = _modal_view(left)
    b, modals_b = _modal_view(right)
    if not modals_a or not modals_b:
        return None
    matcher = SequenceMatcher(None, a, b, autojunk=False)
    if matcher.ratio() < 0.8:
        return None
    for block in matcher.get_matching_blocks():
        for position, modal_a in modals_a.items():
            if not block.a <= position < block.a + block.size:
                continue
            modal_b = modals_b.get(block.b + position - block.a)
            if modal_b and (modal_a["strong"] != modal_b["strong"] or modal_a["negative"] != modal_b["negative"]):
                other = block.b + position - block.a

                def tail(tokens, start):
                    words = []
                    for token in tokens[start + 1:start + 9]:
                        if token == "zzmodalmarkerzz":
                            break
                        words.append(token)
                    return words

                left_tail, right_tail = tail(a, position), tail(b, other)
                if not left_tail or not right_tail or SequenceMatcher(None, left_tail, right_tail, autojunk=False).ratio() < 0.55:
                    continue
                return {
                    "kind": "obligation_negation" if modal_a["negative"] != modal_b["negative"] else "obligation_strength",
                    "modal_a": modal_a["modal"], "modal_b": modal_b["modal"],
                    "negated_a": modal_a["negative"], "negated_b": modal_b["negative"],
                }
    return None


def _potential_conflict(left, right):
    # Cheap gate avoids sentence fitting after a review-volume cap is filled.
    if re.sub(r"\s+", " ", _norm(left)) == re.sub(r"\s+", " ", _norm(right)):
        return False
    profiles = []
    for text in (left, right):
        _, modals = _modal_view(text)
        profiles.append({(m["strong"], m["negative"]) for m in modals.values()})
    return bool(profiles[0] and profiles[1] and any(a != b for a in profiles[0] for b in profiles[1]))


def _numeric_difference(left, right):
    def numbers(text):
        text = re.sub(r"(?m)^\s*(?:\(\d+\)|\d+[.)])(?:[ \t]+|$)", "", _norm(text))
        return re.findall(r"(?<!\w)\d+(?:\.\d+)?(?!\w)", text)
    return numbers(left) != numbers(right)


def _sentence_analysis(left: Article, right: Article, cache, check_conflict):
    for article in (left, right):
        if article["id"] not in cache:
            cache[article["id"]] = _sentences(article["text"])
    sentences = [cache[a["id"]] for a in (left, right)]
    _, matrix = _vectors(sentences[0] + sentences[1])
    chosen = [0, 0]
    conflict = None
    if matrix is not None:
        split = len(sentences[0])
        scores = cosine_similarity(matrix[:split], matrix[split:], dense_output=False).tocsr()
        scores.sort_indices()
        entries = scores.tocoo()
        if entries.nnz:
            best = int(entries.data.argmax())
            chosen = [int(entries.row[best]), int(entries.col[best])]
        if check_conflict:
            candidates = []
            for row in range(len(sentences[0])):
                start, end = scores.indptr[row:row + 2]
                values, columns = scores.data[start:end], scores.indices[start:end]
                for pos in (-values).argsort(kind="stable")[:3]:
                    if values[pos] >= 0.45:
                        candidates.append((float(values[pos]), row, int(columns[pos])))
            for score, i, j in sorted(candidates, key=lambda item: (-item[0], item[1], item[2])):
                change = _obligation_change(sentences[0][i], sentences[1][j])
                if change:
                    # Cite the changed obligation even if a shared boilerplate sentence scored higher.
                    conflict = (change, [i, j])
                    break
    def evidence(indices):
        return [{"article_id": article["id"], "quote": parts[index]}
                for article, parts, index in zip((left, right), sentences, indices)]
    return evidence(chosen), (conflict[0], evidence(conflict[1])) if conflict else None


def _thresholds(config):
    try:
        dup, overlap = float(config.get("dup", 0.85)), float(config.get("overlap", 0.60))
        if not (math.isfinite(dup) and math.isfinite(overlap) and 0 < overlap <= dup <= 1):
            raise ValueError("expected 0 < overlap <= dup <= 1")
        return dup, overlap
    except (TypeError, ValueError) as exc:
        _warn(f"invalid thresholds ({exc}); using dup=0.85, overlap=0.60.")
        return 0.85, 0.60


def _limit(config):
    try:
        value = config.get("max_findings_per_type", 300)
        if isinstance(value, bool) or int(value) != float(value) or int(value) < 0:
            raise ValueError("expected a nonnegative integer")
        return int(value)
    except (TypeError, ValueError, OverflowError):
        _warn("invalid max_findings_per_type; using 300.")
        return 300


def _write_coverage(articles, pairs, conflicts, dup, config):
    path = config.get("coverage_path", "data/coverage.json")
    if path is None:
        return
    groups = defaultdict(list)
    for article in articles:
        groups[article["regulation"]].append(article)
    rows = []
    for reg_a, reg_b in combinations(sorted(groups), 2):
        duplicates, overlaps, matched = 0, 0, set()
        for i, j, score in pairs:
            left, right = articles[i], articles[j]
            if {left["regulation"], right["regulation"]} != {reg_a, reg_b}:
                continue
            matched.add(left["id"] if left["regulation"] == reg_a else right["id"])
            if (i, j) not in conflicts:
                duplicate = score >= dup and not _numeric_difference(left["text"], right["text"])
                duplicates += duplicate
                overlaps += not duplicate
        rows.append({"reg_a": reg_a, "reg_b": reg_b, "articles_a": len(groups[reg_a]),
                     "articles_b": len(groups[reg_b]), "duplicates": duplicates, "overlaps": overlaps,
                     "share_of_a_with_a_match": len(matched) / len(groups[reg_a])})
    try:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        _warn(f"cannot write coverage ({exc}); continuing with findings.")


def _requirement_findings(articles, config):
    candidates = [a for a in articles if not a.get("is_definitions", False)
                  and (a.get("extra") or {}).get("kind") != "glossary_term" and len(a["text"].split()) >= 15]
    dup, overlap = _thresholds(config)
    pairs = sorted(_pair_scores(candidates, overlap, top_k=3), key=lambda pair: (-pair[2], pair[0], pair[1]))
    parents = list(range(len(candidates)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for i, j, score in pairs:
        if (score >= dup and not _numeric_difference(candidates[i]["text"], candidates[j]["text"])
                and not _potential_conflict(candidates[i]["text"], candidates[j]["text"])):
            parents[root(j)] = root(i)
    clusters = defaultdict(list)
    for i, article in enumerate(candidates):
        clusters[root(i)].append(article)
    findings, counts, cache, conflicts = [], defaultdict(int), {}, set()
    limit = _limit(config)
    for i, j, score in pairs:
        left, right = candidates[i], candidates[j]
        numeric_difference = _numeric_difference(left["text"], right["text"])
        duplicate = score >= dup and not numeric_difference
        kind = "duplicate_requirement" if duplicate else "partial_overlap"
        potential = _potential_conflict(left["text"], right["text"])
        if counts[kind] >= limit and (not potential or counts["conflicting_values"] >= limit):
            continue
        evidence, conflict = _sentence_analysis(left, right, cache, potential)
        if conflict:
            change, quotes = conflict
            conflicts.add((i, j))
            if counts["conflicting_values"] < limit:
                strength = change["kind"] == "obligation_strength"
                guidance = strength and any(
                    re.match(r"^(?:AMC|GM)", str(a.get("article_no", "")), re.I)
                    and modal not in _STRONG for a, modal in
                    ((left, change["modal_a"]), (right, change["modal_b"]))
                )
                rationale = (
                    "One text makes it mandatory, the other advisory; the expert decides which is intended."
                    if strength else "The aligned clauses impose opposite obligations; the expert must resolve the difference."
                )
                if guidance:
                    rationale += " The weaker wording appears in AMC/GM guidance, where it may be intentional."
                findings.append(_finding(
                    "conflicting_values", [left, right], quotes,
                    f"Same requirement, different obligation: {_label(left)} and {_label(right)} "
                    f"({change['modal_a']}{' not' if change['negated_a'] else ''} / "
                    f"{change['modal_b']}{' not' if change['negated_b'] else ''})",
                    "harmonize_wording", rationale, score, {**change, "score": score},
                    "medium" if guidance else "high",
                ))
                counts["conflicting_values"] += 1
            # A lexical near-match with a changed obligation is not a duplicate.
            continue
        if counts[kind] >= limit:
            continue
        containment = _containment(left["text"], right["text"])
        extra = {"score": score, "containment": containment}
        if numeric_difference:
            extra["numeric_difference"] = True
        if numeric_difference:
            recommendation = "harmonize_wording"
            rationale = "The articles contain different numeric values or references; review those differences before reusing either requirement."
        elif duplicate:
            cluster = clusters[root(i)]
            common = len({a["regulation"] for a in cluster}) >= 3
            recommendation = "move_to_common" if common else "cross_reference"
            rationale = (
                "This duplicate cluster spans at least three regulations; state the shared requirement once and reference it."
                if common else
                "Same requirement in two regulations; keep it in one (expert chooses the lead) and replace the copy with a reference."
            )
            if common:
                extra["cluster_article_ids"] = sorted(a["id"] for a in cluster)
        elif containment >= 0.8:
            recommendation = "cross_reference"
            rationale = "One article is largely contained in the other; review whether a reference can replace the repeated requirement."
        else:
            recommendation = "harmonize_wording"
            rationale = "The articles share part of a requirement; align the shared wording while preserving any differences in scope."
        regs = {left["regulation"], right["regulation"]}
        if regs == {"CAR-ADR", "CAR-AGA-PART-ADR"} and duplicate:
            recommendation = "cross_reference"
            rationale = "Both CAR IX documents state this requirement. The expert confirms which issue is in force; the other is retired or replaced with a reference."
        elif not numeric_difference and "CAR-BR" in regs and regs.intersection({"CAR-ADR", "CAR-AGA-PART-ADR"}):
            recommendation = "cross_reference"
            rationale = "CAR-BR is the basic regulation; CAR IX should reference it rather than restate it."
        findings.append(_finding(
            kind, [left, right], evidence,
            f"{_label(left)} and {_label(right)} are {'near-identical' if duplicate else 'partially overlapping'} (score {score:.2f})",
            recommendation, rationale, score, extra,
        ))
        counts[kind] += 1
    _write_coverage(candidates, pairs, conflicts, dup, config)
    return findings


def _merge_findings(findings):
    # Several terms may share the same contract-mandated type + article IDs.
    # Keep every term and quote instead of letting pipeline ID dedup discard them.
    merged = {}
    for finding in findings:
        previous = merged.get(finding["id"])
        if previous is None:
            merged[finding["id"]] = finding
            continue
        for evidence in finding["evidence"]:
            if evidence not in previous["evidence"]:
                previous["evidence"].append(evidence)
        terms = previous["extra"].setdefault("terms", [previous["extra"]["term"]])
        details = previous["extra"].setdefault("definitions", [{
            "term": previous["extra"]["term"], "similarity": previous["extra"]["similarity"],
        }])
        term = finding["extra"]["term"]
        if term not in terms:
            terms.append(term)
            previous["summary"] += "; " + finding["summary"]
        details.append({"term": term, "similarity": finding["extra"]["similarity"]})
        if _RANK[finding["severity"]] < _RANK[previous["severity"]]:
            previous["severity"] = finding["severity"]
            previous["recommendation"] = finding["recommendation"]
            previous["rationale"] = finding["rationale"]
        previous["confidence"] = min(previous["confidence"], finding["confidence"])
    return sorted(merged.values(), key=lambda f: (_RANK[f["severity"]], f["type"], f["id"]))


def detect(articles: list[Article], config: dict | None = None) -> list[Finding]:
    """Return review candidates using the shared Finding schema; never edit articles."""
    source = _articles(articles)
    if not source or not _available():
        return []
    config = config or {}
    definitions = _definitions(source)
    findings = _merge_findings(
        _definition_findings(source, definitions, config) + _requirement_findings(source, config)
    )
    counts, kept, limit = defaultdict(int), [], _limit(config)
    for finding in sorted(findings, key=lambda f: (-f["confidence"], f["id"])):
        if counts[finding["type"]] < limit:
            kept.append(finding)
            counts[finding["type"]] += 1
    return sorted(kept, key=lambda f: (_RANK[f["severity"]], f["type"], f["id"]))


def _self_test(mock_articles: list[Article]) -> bool:
    """Acceptance and regression checks live here because P3 owns this file only."""
    import copy
    import tempfile
    import unittest

    detector = detect

    def run(articles, config=None):
        return detector(articles, {"coverage_path": None, **(config or {})})

    def article(reg, text, *, definitions=False, no="1"):
        return {"id": f"{reg}|{no}", "regulation": reg, "article_no": no,
                "text": text, "is_definitions": definitions}

    class SimilarityTests(unittest.TestCase):
        def test_mock_acceptance_and_contract(self):
            original = copy.deepcopy(mock_articles)
            findings = run(mock_articles)
            ids = {a["id"] for a in mock_articles}
            operator = "CAR-DEF|Operator" if "CAR-DEF|Operator" in ids else "CAR-DEF|2"
            occurrence = "CAR-DEF|Occurrence" if "CAR-DEF|Occurrence" in ids else "CAR-DEF|2"
            expected = [
                ({"duplicate_requirement"}, {"CAR-OPS|5", "CAR-AIR|5"}, None),
                ({"partial_overlap", "duplicate_requirement"}, {"CAR-OPS|4", "CAR-AIR|4"}, None),
                ({"divergent_definition"}, {"CAR-OPS|2", operator}, "Operator"),
                ({"duplicate_definition"}, {"CAR-AIR|2", occurrence}, "Occurrence"),
                ({"definition_missing_from_def"}, {"CAR-OPS|2", "CAR-AIR|2"}, "Dry lease"),
            ]
            for types, ids, term in expected:
                self.assertTrue(any(f["type"] in types and set(f["article_ids"]) == ids
                                    and (term is None or f["extra"].get("term") == term) for f in findings),
                                f"Missing {types}: {ids} ({term})")
            sources = {a["id"]: a["text"] for a in mock_articles}
            self.assertEqual(len({f["id"] for f in findings}), len(findings))
            for finding in findings:
                self.assertEqual(finding["id"], finding["type"] + ":" + "+".join(sorted(finding["article_ids"])))
                self.assertFalse(finding["quotes_verified"])
                self.assertTrue(0 <= finding["confidence"] <= 1)
                for evidence in finding["evidence"]:
                    self.assertIn(evidence["quote"], sources[evidence["article_id"]])
            self.assertEqual(mock_articles, original)
            self.assertEqual(findings, run(list(reversed(mock_articles))))

        def test_definition_forms_and_normalization(self):
            source = [
                article("CAR-DEF", '  (a) Operators means a person operating an aircraft.\n'
                        '• المُشَغِّل: الشخص المسؤول عن تشغيل الطائرة بأمان.', definitions=True),
                article("CAR-X", 'For this section, «Operator» shall mean a person operating an aircraft.\n'
                        'Unquoted means this is not a definition article.\n'
                        'المشغل: الشخص المسؤول عن تشغيل الطائرة بأمان.'),
            ]
            index = term_index(source)
            self.assertEqual(len(index["operator"]), 2)
            self.assertEqual(len(index["مشغل"]), 1)
            self.assertNotIn("unquoted", index)
            self.assertEqual(_norm("أَإِآٱ ى ة ـ ١٢"), "اااا ي ه  12")
            self.assertEqual(_term_key('"Safety-records"'), "safetyrecord")
            self.assertEqual(_term_key("الجو"), "الجو")
            self.assertEqual(index["مشغل"][0]["text"], source[0]["text"].splitlines()[1])

        def test_missing_baseline_and_configured_baseline(self):
            text = '"Operator" means a person operating an aircraft.'
            sources = [article("COMMON", text, definitions=True), article("CAR-X", text, definitions=True)]
            self.assertEqual(run(sources), [])
            findings = run(sources, {"def_regulation": "common"})
            self.assertEqual([f["type"] for f in findings], ["duplicate_definition"])
            self.assertEqual(findings[0]["recommendation"], "cross_reference")

        def test_multiple_terms_keep_unique_finding_and_all_quotes(self):
            text = ('"Operator" means a person operating an aircraft.\n'
                    '"Aircraft" means a machine supported in the atmosphere.')
            findings = run([article("CAR-DEF", text, definitions=True),
                               article("CAR-X", text, definitions=True)])
            self.assertEqual(len(findings), 1)
            self.assertEqual(set(findings[0]["extra"]["terms"]), {"Operator", "Aircraft"})
            self.assertEqual(len(findings[0]["evidence"]), 4)

        def test_missing_term_requires_distinct_regulations(self):
            sources = [article("CAR-DEF", '"Other" means an unrelated defined term.', definitions=True)]
            text = '"Dry lease" means an aircraft operated under the lessee certificate.'
            sources += [article("CAR-X", text, definitions=True, no=str(i)) for i in (1, 2)]
            self.assertEqual(run(sources), [])
            sources.append(article("CAR-Y", text, definitions=True))
            findings = run(sources)
            self.assertEqual(len(findings), 1)
            self.assertEqual(len(findings[0]["article_ids"]), 3)

        def test_three_regulation_cluster_and_candidate_filters(self):
            text = ("The operator shall establish, implement and maintain a safety management system "
                    "appropriate to the size and complexity of all its aircraft operations.")
            sources = [article(reg, text) for reg in ("CAR-A", "CAR-B", "CAR-C")]
            findings = run(sources)
            self.assertEqual(len(findings), 3)
            self.assertTrue(all(f["recommendation"] == "move_to_common" for f in findings))
            self.assertTrue(all(len(f["extra"]["cluster_article_ids"]) == 3 for f in findings))
            same_reg = [article("CAR-A", text), article("CAR-A", text, no="2")]
            self.assertEqual(run(same_reg), [])
            self.assertEqual(similar_pairs(same_reg), [])
            short = [article("CAR-A", "A short identical requirement."), article("CAR-B", "A short identical requirement.")]
            self.assertEqual(run(short), [])
            self.assertEqual(len(similar_pairs(short, 1.0)), 1)

        def test_overlap_containment_and_sentence_evidence(self):
            base = ("The operator shall retain all aircraft maintenance records for a period of two years "
                    "after the aircraft has been permanently withdrawn from service.")
            unrelated = "Bananas grow in tropical orchards far from the northern ice fields."
            sources = [article("CAR-A", unrelated + "\n" + base), article("CAR-B", base)]
            finding = run(sources, {"dup": 1.0, "overlap": 0.2})[0]
            self.assertEqual(finding["type"], "partial_overlap")
            self.assertEqual(finding["recommendation"], "cross_reference")
            self.assertEqual(finding["extra"]["containment"], 1.0)
            self.assertTrue(all(ev["quote"] == base for ev in finding["evidence"]))

        def test_empty_input_and_pair_order(self):
            self.assertEqual(run([]), [])
            self.assertEqual(term_index([]), {})
            self.assertEqual(similar_pairs([]), [])
            self.assertEqual(run([article("CAR-A", "   ")]), [])
            pairs = similar_pairs(mock_articles)
            self.assertEqual(pairs, sorted(pairs, key=lambda pair: (-pair[2], pair[0], pair[1])))
            self.assertTrue(all(a < b for a, b, _ in pairs))
            self.assertEqual(pairs, similar_pairs(list(reversed(mock_articles))))

        def test_pair_cap_and_arabic_requirements(self):
            text = "يجب على المشغل الاحتفاظ بجميع سجلات صيانة الطائرات لمدة خمس سنوات بعد سحب الطائرة نهائيا من الخدمة."
            sources = [article(f"CAR-{i:03}", text) for i in range(65)]
            self.assertEqual(len(similar_pairs(sources)), 2000)
            finding = run(sources[:2])[0]
            self.assertEqual(finding["type"], "duplicate_requirement")
            self.assertTrue(all(ev["quote"] == text for ev in finding["evidence"]))

        def test_glossary_curly_quotes_aliases_and_pdf_wraps(self):
            body = "A defined area intended to accommodate aircraft for loading and unloading of passengers and cargo."
            glossary = {**article("CAR-DEF", body, definitions=True, no="Apron (APN)"),
                        "title": "Apron (APN)", "extra": {"kind": "glossary_term"}}
            local = article("CAR-BR", '(12) ‘apron’ means A defined area intended to accommodate aircraft\n'
                            'for loading and unloading of passengers and cargo.\n'
                            '(13) ‘certificate’ means any certificate or approval issued by the Authority;', definitions=True, no="3")
            index = term_index([glossary, local])
            self.assertEqual(len(index["apron"]), 2)
            self.assertEqual(index["apn"][0]["alias_of"], "apron")
            self.assertEqual(_term_key("aircraft operato r"), _term_key("aircraft operator"))
            findings = run([glossary, local])
            self.assertEqual(len(findings), 1)
            self.assertEqual(findings[0]["type"], "duplicate_definition")
            self.assertIn("\n", findings[0]["evidence"][0]["quote"] if findings[0]["evidence"][0]["article_id"] == local["id"] else findings[0]["evidence"][1]["quote"])
            sources = {a["id"]: a["text"] for a in [glossary, local]}
            for evidence in findings[0]["evidence"]:
                self.assertIn(evidence["quote"], sources[evidence["article_id"]])
            alias = article("CAR-X", '"APN" means ' + body, definitions=True)
            finding = run([glossary, alias])[0]
            self.assertTrue(finding["extra"]["alias_used"])

        def test_glossary_see_links_are_indexed_without_comparisons(self):
            glossary = {**article("CAR-DEF", "See: Aircraft operator", definitions=True, no="Operator"),
                        "title": "Operator", "extra": {"kind": "glossary_term"}}
            local = article("CAR-BR", '‘Operator’ means a person operating an aircraft.', definitions=True)
            self.assertEqual(term_index([glossary])["operator"][0]["see_also"], "Aircraft operator")
            self.assertEqual(run([glossary, local]), [])
            another = {**local, "id": "CAR-X|1", "regulation": "CAR-X"}
            self.assertEqual(run([glossary, local, another]), [])

        def test_different_definitions_inside_car_def(self):
            sources = [{**article("CAR-DEF", text, definitions=True, no=term), "title": term,
                        "extra": {"kind": "glossary_term"}} for term, text in
                       [("Operator", "A person operating an aircraft."),
                        ("Operator (OPR)", "A company managing the safety of an aerodrome.")]]
            finding = run(sources)[0]
            self.assertEqual(finding["type"], "divergent_definition")
            self.assertEqual(finding["severity"], "medium")

        def test_low_similarity_respects_sector_specific_definitions(self):
            sources = [article("CAR-DEF", '"Operator" means a person flying aircraft in commercial transport.', definitions=True),
                       article("CAR-X", '"Operator" means whoever controls financial investments and manages banking accounts.', definitions=True)]
            finding = run(sources)[0]
            self.assertLess(finding["extra"]["similarity"], 0.5)
            self.assertEqual(finding["severity"], "high")
            self.assertEqual(finding["recommendation"], "keep_with_justification")

        def test_obligation_strength_negation_and_guidance(self):
            base = ("Initial notification of a vacancy shall, within 5 working days, be submitted to the Authority "
                    "by the aerodrome operator with all relevant details.")
            for changed, kind in [(base.replace("shall", "should"), "obligation_strength"),
                                  (base.replace("shall", "shall not"), "obligation_negation")]:
                sources = [article("CAR-ADR", base, no="4.7"),
                           article("CAR-AGA-PART-ADR", changed, no="ADR.OR.D.015")]
                findings = run(sources)
                self.assertEqual(len(findings), 1)
                self.assertEqual(findings[0]["type"], "conflicting_values")
                self.assertEqual(findings[0]["extra"]["kind"], kind)
                self.assertEqual(findings[0]["severity"], "high")
            sources[1]["text"] = base.replace("shall", "should")
            sources[1]["article_no"] = "AMC1 ADR.OR.D.015"
            sources[1]["id"] = "CAR-AGA-PART-ADR|AMC1 ADR.OR.D.015"
            finding = run(sources)[0]
            self.assertEqual(finding["severity"], "medium")
            self.assertIn("guidance", finding["rationale"])

        def test_modal_change_is_not_hidden_by_identical_boilerplate(self):
            shared = "The operator shall retain all records relevant to the continuing safety of aircraft operations for inspection. "
            change = "The operator shall report each vacancy within five working days to the Authority."
            sources = [article("CAR-A", shared + change), article("CAR-B", shared + change.replace("shall", "should"))]
            findings = run(sources)
            self.assertEqual([f["type"] for f in findings], ["conflicting_values"])
            self.assertTrue(all("vacancy" in ev["quote"] for ev in findings[0]["evidence"]))
            self.assertIsNone(_obligation_change("The operator shall report occurrences.", "The mechanic should check engine temperature."))

        def test_top_matches_volume_limits_and_coverage(self):
            text = "The operator shall establish implement and maintain a safety management system appropriate to the size and complexity of all aircraft operations."
            sources = [article(reg, text, no=str(i)) for reg in ("CAR-A", "CAR-B") for i in range(20)]
            with tempfile.TemporaryDirectory(prefix="reglens-p3-test-") as directory:
                path = Path(directory) / "coverage.json"
                findings = run(sources, {"max_findings_per_type": 7, "coverage_path": str(path)})
                self.assertEqual(len(findings), 7)
                coverage = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(coverage[0]["articles_a"], 20)
                self.assertEqual(coverage[0]["share_of_a_with_a_match"], 1.0)
                self.assertLess(coverage[0]["duplicates"], 400)
                self.assertGreater(coverage[0]["duplicates"], 7)
            self.assertEqual(run(sources, {"max_findings_per_type": 0}), [])

        def test_wrapped_requirement_quotes_and_empty_bodies(self):
            base = "The operator shall retain all aircraft maintenance records\nfor five years after the aircraft is permanently withdrawn from service."
            sources = [article("CAR-A", base), article("CAR-B", base.replace("\n", " "))]
            finding = run(sources)[0]
            self.assertEqual(finding["type"], "duplicate_requirement")
            self.assertEqual(finding["evidence"][0]["quote"], base)
            self.assertEqual(run([article("CAR-A", None), article("CAR-B", "")]), [])

        def test_numeric_changes_require_review_even_above_duplicate_threshold(self):
            base = "The operator shall retain all aircraft maintenance records for 2 years after the aircraft has been permanently withdrawn from service."
            finding = run([article("CAR-A", base), article("CAR-B", base.replace("2 years", "3 years"))])[0]
            self.assertGreater(finding["extra"]["score"], 0.85)
            self.assertEqual(finding["type"], "partial_overlap")
            self.assertEqual(finding["recommendation"], "harmonize_wording")
            self.assertTrue(finding["extra"]["numeric_difference"])

        def test_definition_colon_does_not_create_a_second_term(self):
            source = article("CAR-DEF", '“Operator” means a person: someone who operates aircraft.', definitions=True)
            self.assertEqual(list(term_index([source])), ["operator"])

        def test_arabic_modal_change(self):
            text = "يجب على المشغل الاحتفاظ بجميع سجلات صيانة الطائرات لمدة خمس سنوات بعد سحب الطائرة نهائيا من الخدمة."
            finding = run([article("CAR-A", text), article("CAR-B", text.replace("يجب", "ينبغي"))])[0]
            self.assertEqual(finding["type"], "conflicting_values")
            self.assertEqual(finding["extra"]["kind"], "obligation_strength")

        def test_medium_similarity_and_optional_config_fallback(self):
            source = [article("CAR-DEF", '"Operator" means a person operating an aircraft for commercial passenger transport.', definitions=True),
                      article("CAR-X", '"Operator" means a person operating an aircraft for private passenger transport.', definitions=True)]
            finding = run(source)[0]
            self.assertTrue(0.5 <= finding["extra"]["similarity"] < 0.8)
            self.assertEqual(finding["severity"], "medium")
            self.assertEqual(finding["recommendation"], "harmonize_wording")
            self.assertEqual(_thresholds({"dup": "bad"}), (0.85, 0.60))
            self.assertEqual(_limit({"max_findings_per_type": -1}), 300)

        def test_modals_must_align_to_the_same_predicate(self):
            left = "The safety Post Holder and quality Post Holder should be senior management positions and should not report to the operations Post Holder."
            right = "The Safety Post Holder and Quality Post Holder should not report to the Operations Post Holder."
            self.assertIsNone(_obligation_change(left, right))
            self.assertIsNotNone(_obligation_change(left, left.replace("should not report", "should report")))

        def test_evidence_avoids_headings_and_conjunctions(self):
            text = "ATTENDANCE AT AUDITS\n(a) The operator shall provide access to all records and documents relevant to the inspection of the aircraft.\nand\n(b) The operator shall assist the inspectors during all audits."
            finding = run([article("CAR-A", text), article("CAR-B", text.replace("all audits", "periodic audits"))])[0]
            self.assertTrue(all("shall" in ev["quote"] for ev in finding["evidence"]))
            self.assertTrue(all(len(ev["quote"].split()) >= 5 for ev in finding["evidence"]))

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SimilarityTests)
    return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("articles", nargs="?", default="data/mock_articles.json")
    parser.add_argument("--config", help="JSON file with dup, overlap and/or def_regulation")
    parser.add_argument("--test", action="store_true", help="run the mock acceptance and regression checks")
    parser.add_argument("--stats", action="store_true", help="print counts, CAR IX matches and obligation conflicts")
    args = parser.parse_args(argv)
    # Windows terminals may not support Arabic or smart quotes in source evidence.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        articles = json.loads(Path(args.articles).read_text(encoding="utf-8-sig"))
        if not isinstance(articles, list) or not all(isinstance(a, dict) for a in articles):
            raise ValueError("articles JSON must contain a list of article objects")
    except (OSError, ValueError) as exc:
        _warn(f"cannot read input: {exc}")
        return 1 if args.test else 0
    config = {}
    if args.config:
        try:
            config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
            if not isinstance(config, dict):
                raise ValueError("config JSON must contain an object")
        except (OSError, ValueError) as exc:
            _warn(f"cannot read optional config ({exc}); using defaults.")
            config = {}
    findings = detect(articles, config)
    if args.stats:
        from collections import Counter

        print(json.dumps({"articles": len(articles), "findings": len(findings),
                          "by_type": dict(Counter(f["type"] for f in findings)),
                          "thresholds": dict(zip(("dup", "overlap"), _thresholds(config)))}, indent=2))
        source = [a for a in _articles(articles) if not a.get("is_definitions", False)
                  and (a.get("extra") or {}).get("kind") != "glossary_term" and len(a["text"].split()) >= 15]
        print("\nTOP 20 CAR-ADR / CAR-AGA-PART-ADR requirement pairs:")
        if _available():
            pairs = ((i, j, score) for i, j, score in _pair_scores(source, 0.25, top_k=3)
                     if {source[i]["regulation"], source[j]["regulation"]} == {"CAR-ADR", "CAR-AGA-PART-ADR"})
            for i, j, score in heapq.nsmallest(20, pairs, key=lambda pair: (-pair[2], pair[0], pair[1])):
                print(f"  {score:.4f}  {source[i]['id']} + {source[j]['id']}")
                for article in (source[i], source[j]):
                    print("    " + re.sub(r"\s+", " ", article["text"])[:100])
        print("\nObligation changes:")
        print(json.dumps([f for f in findings if f["extra"].get("kind") in
                          {"obligation_strength", "obligation_negation"}], ensure_ascii=False, indent=2))
    else:
        print(json.dumps(findings, ensure_ascii=False, indent=2))
        print("\nTOP 20 cross-regulation pairs (lexical similarity):")
        for left, right, score in similar_pairs(articles)[:20]:
            print(f"  {score:.4f}  {left} + {right}")
    if args.test:
        if not _available():
            return 1
        return 0 if _self_test(articles) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
