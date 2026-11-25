"""Detect duration conflicts, broken CAR references, and ICAO edition issues."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from urllib.parse import quote as encode_id

Article = dict
Finding = dict
Reference = dict

_TRANSLATION = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹أإآٱىة",
                             "01234567890123456789اااايه")
_DIACRITICS = re.compile(r"[\u064b-\u0652\u0670\u0640]")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?؟؛;])\s+|\n+")
_CITATION_ABBREVIATION = re.compile(r"(?:\bart\.|\bedition\s+no\.)$", re.I)

_NUMBER_WORDS = dict(zip(
    "one two three four five six seven eight nine ten twelve fifteen thirty sixty ninety".split(),
    (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 30, 60, 90),
))
_UNIT_ALIASES = {
    "hour": ("hour", "hours", "ساعه", "ساعات"),
    "day": ("day", "days", "يوما", "يوم", "ايام", "calendar day", "calendar days"),
    "working day": ("working day", "working days", "business day", "business days",
                    "ايام عمل", "يوم عمل"),
    "week": ("week", "weeks", "اسبوعا", "اسبوع", "اسابيع"),
    "month": ("month", "months", "شهرا", "شهر", "اشهر", "شهور"),
    "year": ("year", "years", "سنه", "سنوات", "عاما", "عام", "اعوام"),
}
_CANONICAL_UNIT = {alias: unit for unit, aliases in _UNIT_ALIASES.items() for alias in aliases}
_HOURS = {"hour": 1, "day": 24, "working day": 24, "week": 168, "month": 730, "year": 8760}
_UNITS_PATTERN = "|".join(
    re.escape(alias).replace(r"\ ", r"\s+")
    for alias in sorted(_CANONICAL_UNIT, key=len, reverse=True)
)
_VALUE = re.compile(
    rf"(?<![\w.])\(?\s*(\d+(?:\.\d+)?|{'|'.join(_NUMBER_WORDS)})"
    rf"\s*\)?\s*-?\s*({_UNITS_PATTERN})(?!\w)"
)
_TOPICS = {
    "retention": r"retain|retention|\bkept\b|\bkeep\b|preserv|archiv|احتفاظ|يحتفظ|حفظ",
    "renewal": r"renew|revalidat|تجديد|يجدد|تجدد",
    "reporting": r"report|notif|\binform|submit|ابلاغ|يبلغ|اخطار|يخطر|تقرير",
    "validity": r"valid|expir|صلاحيه|ساري|صالح|انتهاء",
    "training_interval": r"recurrent|refresher|training|تدريب",
}
_OBJECTS = {
    "dangerous_goods": r"dangerous\s+goods|بضائع\s+خطره|مواد\s+خطره",
    "occurrence": r"occurrence|\bincident|\baccident|حادث|حوادث|واقعه",
    "maintenance": r"maintenance|صيانه",
    "medical": r"medical|طبي",
    "fuel": r"\bfuel\b|وقود",
    "certificate": r"licen|certificat|approval|رخص|ترخيص|شهاده|اعتماد",
    "training": r"training|تدريب",
    "security": r"security|امن",
    "flight": r"\bflight\b|journey|رحله",
}

_ARTICLE_NUMBER = r"(?:[A-Z]{1,6}\.)*\d+(?:\.[A-Z0-9]+)*(?:-\d+)*[A-Z]?"
_EXTERNAL = re.compile(
    rf"\bCAR[\s\-–]+([A-Z0-9]{{1,10}})\b"
    rf"(?:[\s,]*(?:Article|Art\.?|Section|§|المادة|مادة)?\s*\(?\s*"
    rf"({_ARTICLE_NUMBER})(?!\w))?"
)
_INTERNAL = re.compile(
    r"\b(?:article|art\.?|الماده|ماده)\s*\(?\s*(\d+(?:\.\d+)*)\s*\)?\s*"
    r"(?:of this (?:regulation|part)|من هذا النظام|من هذه اللائحه|من هذا الجزء)\b"
)
_DOCUMENT = re.compile(
    r"\b(?:icao\s+)?(annex|ملحق|الملحق)\s*\(?\s*(\d+)\b|\bdoc\s*(\d{4})\b"
)
_ARABIC_ORDINALS = {
    "الاول": 1, "الاولي": 1, "الحادي": 1, "الحاديه": 1,
    "الثاني": 2, "الثانيه": 2, "الثالث": 3, "الثالثه": 3, "الرابع": 4, "الرابعه": 4,
    "الخامس": 5, "الخامسه": 5, "السادس": 6, "السادسه": 6, "السابع": 7, "السابعه": 7,
    "الثامن": 8, "الثامنه": 8, "التاسع": 9, "التاسعه": 9, "العاشر": 10, "العاشره": 10,
}
_EDITION = re.compile(
    r"\b(\d+)\s*(?:st|nd|rd|th)?\s*edition\b"
    r"|\bedition\s*(?:no\.?\s*)?(\d+)\b"
    rf"|\b(?:الطبعه|الاصدار)\s*(?:رقم\s*)?\(?\s*"
    rf"(?:(\d+)|({'|'.join(_ARABIC_ORDINALS)})(?:\s+(عشره?))?)\b"
)


def norm(text: str) -> str:
    """Normalize text for matching only."""
    normalized = unicodedata.normalize("NFKC", text).lower().translate(_TRANSLATION)
    return _DIACRITICS.sub("", normalized)


def _sentences(text: str):
    start = 0
    for boundary in _SENTENCE_BREAK.finditer(text):
        # A reference marker or edition abbreviation can contain a sentence-ending dot.
        if (_CITATION_ABBREVIATION.search(text[start:boundary.start()])
                and re.match(r"\(?\s*(?:\d|[A-Z]{1,6}\.)", text[boundary.end():])):
            continue
        sentence = text[start:boundary.start()].strip()
        if sentence:
            yield sentence
        start = boundary.end()
    sentence = text[start:].strip()
    if sentence:
        yield sentence


def _label(article: Article) -> str:
    return f"{article['regulation']} Art.{article['article_no']}"


def _finding(kind: str, severity: str, article_ids: list[str],
             evidence: list[tuple[str, str]], summary: str, recommendation: str,
             rationale: str, confidence: float, extra: dict) -> Finding:
    return {
        "id": f"{kind}:{'+'.join(sorted(article_ids))}",
        "type": kind,
        "severity": severity,
        "article_ids": sorted(article_ids),
        "evidence": [{"article_id": article_id, "quote": sentence}
                     for article_id, sentence in dict.fromkeys(evidence)],
        "summary": summary,
        "recommendation": recommendation,
        "rationale": rationale,
        "detector": "rules",
        "confidence": confidence,
        "quotes_verified": False,
        "extra": extra,
    }


def _value_facts(sentence: str):
    topic = next((name for name, pattern in _TOPICS.items() if re.search(pattern, sentence)), None)
    if topic is None:
        return
    obj = next((name for name, pattern in _OBJECTS.items() if re.search(pattern, sentence)), "general")
    for match in _VALUE.finditer(sentence):
        token, unit_token = match.groups()
        number = _NUMBER_WORDS.get(token)
        if number is None:
            amount = token.rstrip("0").rstrip(".") if "." in token else token
        else:
            amount = str(number)
        value = Fraction(amount)
        unit = _CANONICAL_UNIT[" ".join(unit_token.split())]
        display_unit = "day" if unit == "working day" else unit
        plural = "" if value == 1 else "s"
        qualifier = " (working)" if unit == "working day" else ""
        yield topic, obj, value * _HOURS[unit], f"{amount} {display_unit}{plural}{qualifier}"


def _duration_findings(sentences: dict, by_id: dict):
    groups = defaultdict(lambda: defaultdict(list))
    for article_id, rows in sentences.items():
        for original, normalized in rows:
            for topic, obj, hours, display in _value_facts(normalized):
                groups[topic, obj][article_id].append((hours, display, original))

    for (topic, obj), group in sorted(groups.items()):
        values = {article_id: {fact[0] for fact in facts} for article_id, facts in group.items()}
        involved = set()
        for left, right in combinations(sorted(values), 2):
            if values[left].isdisjoint(values[right]):
                involved.update((left, right))
        if not involved:
            continue
        ids = sorted(involved)
        displays = {article_id: list(dict.fromkeys(fact[1] for fact in group[article_id]))
                    for article_id in ids}
        evidence = [(article_id, fact[2]) for article_id in ids for fact in group[article_id]]
        details = "; ".join(f"{_label(by_id[aid])} = {', '.join(displays[aid])}" for aid in ids)
        title = f"{obj.replace('_', ' ')} {topic.replace('_', ' ')}".capitalize()
        regulations = {by_id[aid]["regulation"] for aid in ids}
        yield _finding(
            "conflicting_values", "medium" if obj == "general" else "high", ids, evidence,
            f"{title}: {details}", "move_to_common" if len(regulations) >= 3 else "harmonize_wording",
            "The provisions specify different durations for a similar obligation. "
            "An expert can confirm the scope and choose a common value where appropriate.",
            0.6 if obj == "general" else 0.9, {"topic": topic, "object": obj, "values": displays},
        )


def _regulation_code(regulation: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", regulation.upper().removeprefix("CAR"))


def _article_number(number: str, *, strip_prefix: bool = False) -> str:
    number = re.sub(r"\s+", "", norm(number)).upper()
    if strip_prefix:
        number = re.sub(r"^(?:[A-Z]{1,6}\.)+(?=\d)|^[A-Z]{2,6}(?=\d)", "", number)
    return number


def _reference_rows(sentences: dict, by_id: dict):
    regulations = defaultdict(set)
    article_ids = defaultdict(lambda: defaultdict(list))
    for article_id, article in by_id.items():
        name = article["regulation"]
        code = _regulation_code(name)
        regulations[code].add(name)
        exact = _article_number(article["article_no"])
        shorthand = _article_number(article["article_no"], strip_prefix=True)
        for number in {exact, shorthand}:
            article_ids[name][number].append(article_id)

    for source_id, rows in sentences.items():
        own_regulation = by_id[source_id]["regulation"]
        for original, normalized in rows:
            # Preserve the original capitalization required by the external CAR pattern.
            targets = [(f"CAR-{match[1]}", _article_number(match[2]) if match[2] else None, False)
                       for match in _EXTERNAL.finditer(original)]
            targets.extend((own_regulation, match[1], True) for match in _INTERNAL.finditer(normalized))
            seen = set()
            for target, number, internal in targets:
                code = _regulation_code(target)
                if internal:
                    names = [own_regulation]
                elif code in regulations:
                    names = sorted(regulations[code])
                else:
                    names = sorted(name for loaded, names in regulations.items()
                                   if loaded.startswith(code) for name in names)
                exact = [aid for name in names for aid in article_ids[name].get(number, [])
                         if _article_number(by_id[aid]["article_no"]) == number]
                matches = exact
                if not exact and number is not None:
                    shorthand = _article_number(number, strip_prefix=True)
                    matches = [aid for name in names for aid in article_ids[name].get(shorthand, [])]
                resolved = matches[0] if len(matches) == 1 else None
                if resolved is not None:
                    target = by_id[resolved]["regulation"]
                elif len(names) == 1:
                    target = names[0]
                key = target, number
                if key in seen:
                    continue
                seen.add(key)
                row = {"from": source_id, "to_regulation": target, "to_article": number,
                       "resolved_id": resolved, "quote": original}
                yield row, names, matches


def _reference_findings(rows, by_id: dict):
    issues = defaultdict(list)
    for row, names, matches in rows:
        if row["resolved_id"] is not None:
            continue
        if not names or len(matches) > 1:
            kind = "unverifiable_reference"
        elif row["to_article"] is not None:
            kind = "broken_reference"
        else:
            continue
        key = kind, row["from"], row["to_regulation"], row["to_article"]
        issues[key].append((row["quote"], bool(matches)))

    totals = Counter((kind, source) for kind, source, _, _ in issues)
    for (kind, source, target, number), citations in sorted(
        issues.items(), key=lambda item: (*item[0][:3], item[0][3] or "")
    ):
        citation = f"{target} Article {number}" if number is not None else target
        summary = f"{_label(by_id[source])} refers to {citation}"
        broken = kind == "broken_reference"
        if broken:
            summary += ", which does not exist in the loaded regulation"
            rationale = "Target may have been repealed, moved or renumbered."
        elif citations[0][1]:
            summary += ", which matches multiple loaded articles"
            rationale = "The target is ambiguous in the analysed sample. Verify it manually."
        else:
            summary += ", which could not be verified"
            rationale = "Target regulation is not in the analysed sample. Verify it manually."
        finding = _finding(
            kind, "high" if broken else "low", [source], [(source, line) for line, _ in citations],
            summary, "correct_reference" if broken else "keep_with_justification", rationale,
            0.95 if broken else 0.6, {"to_regulation": target, "to_article": number},
        )
        if totals[kind, source] > 1:
            finding["id"] += "+" + encode_id(f"{target}|{number or ''}", safe="|-")
        yield finding


def _edition_number(text: str) -> int | None:
    match = _EDITION.search(text)
    if match is None:
        return None
    number = match[1] or match[2] or match[3]
    if number:
        return int(number)
    return _ARABIC_ORDINALS[match[4]] + (10 if match[5] else 0)


def _ordinal(number: int) -> str:
    suffix = "th"
    if not 11 <= number % 100 <= 13:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def _icao_findings(sentences: dict, by_id: dict, current_editions: dict):
    citations = defaultdict(set)
    for article_id, rows in sentences.items():
        for original, normalized in rows:
            mentions = list(_DOCUMENT.finditer(normalized))
            for index, mention in enumerate(mentions):
                end = mention.end() + 100
                if index + 1 < len(mentions):
                    end = min(end, mentions[index + 1].start())
                edition = _edition_number(normalized[mention.end():end])
                if edition:
                    doc = f"Doc {mention[3]}" if mention[3] else f"Annex {int(mention[2])}"
                    citations[doc].add((edition, article_id, original))

    for document, entries in sorted(citations.items()):
        editions = sorted({edition for edition, _, _ in entries})
        current = current_editions.get(document)
        if isinstance(current, str) and re.fullmatch(r"[0-9]+", current):
            current = int(current)
        outdated = sorted(
            {aid for edition, aid, _ in entries if edition < current}
        ) if type(current) is int and current > 0 else []
        if len(editions) < 2 and not outdated:
            continue
        ordered = sorted(entries, key=lambda entry: (entry[1], entry[0], entry[2]))
        ids = sorted({aid for _, aid, _ in ordered})
        support = defaultdict(set)
        for edition, aid, _ in ordered:
            support[edition].add(_label(by_id[aid]))
        details = "; ".join(f"{_ordinal(edition)} edition ({', '.join(sorted(support[edition]))})"
                            for edition in editions)
        summary = f"ICAO {document} cited as {details}"
        rationale = "Confirm the intended edition with the regulatory expert and align the citations."
        if outdated:
            outdated_labels = ", ".join(_label(by_id[aid]) for aid in outdated)
            summary += f". Outdated edition cited by {outdated_labels} (configured: {_ordinal(current)})"
            rationale = "Check the applicable edition and correct citations older than the configured edition."
        extra = {"document": document, "editions": editions, "outdated_article_ids": outdated}
        if type(current) is int and current > 0:
            extra["current_edition"] = current
        yield _finding(
            "icao_edition_issue", "high", ids, [(aid, line) for _, aid, line in ordered], summary,
            "correct_reference" if outdated else "harmonize_wording", rationale, 0.95, extra,
        )


def _article_sentences(articles: list[Article]) -> tuple[dict, dict]:
    by_id = {article["id"]: article for article in articles}
    sentences = {aid: [(sentence, norm(sentence)) for sentence in _sentences(by_id[aid]["text"])]
                 for aid in sorted(by_id)}
    return by_id, sentences


def references(articles: list[Article]) -> list[Reference]:
    """Return detected CAR citations, with resolved article IDs where available."""
    by_id, sentences = _article_sentences(articles)
    return [row for row, _, _ in _reference_rows(sentences, by_id)]


def detect(articles: list[Article], config: dict | None = None) -> list[Finding]:
    """Build rule findings in the RegLens contract format."""
    by_id, sentences = _article_sentences(articles)
    current_editions = (config or {}).get("icao_current")
    if not isinstance(current_editions, dict):
        current_editions = {}
    findings = [
        *_duration_findings(sentences, by_id),
        *_reference_findings(_reference_rows(sentences, by_id), by_id),
        *_icao_findings(sentences, by_id, current_editions),
    ]
    totals = Counter(finding["id"] for finding in findings)
    for finding in findings:
        if totals[finding["id"]] > 1:
            extra = finding["extra"]
            key = extra.get("document") or f"{extra['topic']}:{extra['object']}"
            finding["id"] += "+" + encode_id(key, safe="")
    return sorted(findings, key=lambda finding: finding["id"])


def _self_test(articles: list[Article]) -> None:
    findings = detect(articles)
    assert Counter(f["type"] for f in findings) == {
        "conflicting_values": 2, "broken_reference": 2, "icao_edition_issue": 1,
    }
    durations = {f["extra"]["object"]: f for f in findings if f["type"] == "conflicting_values"}
    assert durations["maintenance"]["extra"]["values"] == {
        "CAR-AIR|4": ["3 years"], "CAR-AOC|2": ["5 years"], "CAR-OPS|4": ["2 years"],
    }
    assert durations["maintenance"]["extra"]["topic"] == "retention"
    assert durations["maintenance"]["recommendation"] == "move_to_common"
    assert durations["occurrence"]["extra"]["values"] == {
        "CAR-AIR|3": ["24 hours"], "CAR-OPS|3": ["72 hours"],
    }
    assert {(r["from"], r["to_regulation"], r["to_article"], r["resolved_id"])
            for r in references(articles)} == {
        ("CAR-AIR|6", "CAR-OPS", "9", None), ("CAR-OPS|6", "CAR-OPS", "14", None),
    }
    edition = next(f for f in findings if f["type"] == "icao_edition_issue")
    assert edition["extra"]["document"] == "Annex 6"
    assert edition["extra"]["editions"] == [10, 12]
    by_id = {article["id"]: article["text"] for article in articles}
    assert all(ev["quote"] in by_id[ev["article_id"]] for f in findings for ev in f["evidence"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("articles", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent / "data" / "mock_articles.json")
    parser.add_argument("--test", action="store_true", help="check the shared mock fixture")
    args = parser.parse_args()
    try:
        articles = json.loads(args.articles.read_text(encoding="utf-8-sig"))
        if not isinstance(articles, list) or any(
            not isinstance(article, dict)
            or not all(isinstance(article.get(key), str)
                       for key in ("id", "regulation", "article_no", "text"))
            for article in articles
        ):
            raise ValueError("Expected a JSON list of Article objects.")
    except (OSError, ValueError) as error:
        print(f"[rules] Warning: {error}", file=sys.stderr)
        return 1
    if args.test:
        _self_test(articles)
        print("P2 self-test passed.")
    else:
        findings = detect(articles)
        payload = json.dumps(findings, ensure_ascii=False, indent=2)
        try:
            payload.encode(sys.stdout.encoding or "utf-8")
        except UnicodeEncodeError:
            # Keep redirected JSON readable by Windows shells with a legacy code page.
            payload = json.dumps(findings, indent=2)
        print(payload)
        counts = Counter(f["type"] for f in findings)
        print(f"[rules] {len(findings)} findings by type: "
              f"{json.dumps(dict(sorted(counts.items())))}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
