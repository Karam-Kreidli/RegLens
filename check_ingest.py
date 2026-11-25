"""P1 sanity checks for data/articles.json on the real CAR I + CAR IX files.

Usage: python check_ingest.py [data/articles.json]
Prints PASS / WARN / FAIL per check and the articles to look at by hand. Exit code 1 if any FAIL.
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter

EXPECTED = ["CAR-DEF", "CAR-BR", "CAR-AGA-PART-ADR"]  # + CAR-ADR if Issue 11 is loaded
results = []


def report(status, name, detail=""):
    results.append(status)
    print(f"[{status}] {name}" + (f": {detail}" if detail else ""))


def main(path):
    with open(path, encoding="utf-8") as fh:
        arts = json.load(fh)
    by_reg = {}
    for a in arts:
        by_reg.setdefault(a["regulation"], []).append(a)
    ids = {a["id"] for a in arts}

    print(f"{len(arts)} articles in {path}")
    for reg, items in sorted(by_reg.items()):
        print(f"  {reg:<22} {len(items):>5}")
    print()

    missing = [r for r in EXPECTED if r not in by_reg]
    report("FAIL" if missing else "PASS", "all 4 regulations present", ", ".join(missing) or "")
    report("PASS" if len(ids) == len(arts) else "FAIL", "ids unique", f"{len(arts) - len(ids)} duplicates")

    # CAR-DEF glossary
    d = by_reg.get("CAR-DEF", [])
    gl = [a for a in d if (a.get("extra") or {}).get("kind") == "glossary_term"]
    report("PASS" if 1300 <= len(gl) <= 1600 else "WARN", "CAR-DEF parsed as glossary", f"{len(gl)} terms (expected ~1,440)")
    for term, start in [("Aerodrome certificate", "A certificate issued by the Authority under Civil Aviation Regulation Part IX"),
                        ("Control zone (CTR)", "A controlled airspace extending upwards from the surface of the earth"),
                        ("Apron (APN)", None)]:
        a = next((x for x in gl if x["title"].lower() == term.lower()), None)
        if not a:
            report("WARN", f"CAR-DEF term '{term}'", "not found (check how the term was split)")
        elif start and not re.sub(r"\s+", " ", a["text"]).startswith(start):
            report("WARN", f"CAR-DEF term '{term}'", f"definition starts: {a['text'][:90]!r}")
        else:
            report("PASS", f"CAR-DEF term '{term}'", a["text"][:70])
    junk = [a for a in gl if len(a["title"].split()) > 8]
    report("PASS" if len(junk) < 10 else "WARN", "CAR-DEF junk rows (term > 8 words)", f"{len(junk)}")

    # CAR-BR
    br = {a["article_no"] for a in by_reg.get("CAR-BR", [])}
    want = {str(i) for i in range(1, 21)}
    report("PASS" if want <= br else "WARN", "CAR-BR Articles 1-20", "missing " + ", ".join(sorted(want - br, key=int)) if want - br else "")
    a3 = next((a for a in by_reg.get("CAR-BR", []) if a["article_no"] == "3"), None)
    report("PASS" if a3 and a3["is_definitions"] and "means" in a3["text"] else "WARN", "CAR-BR Art. 3 is the definitions article")

    # CAR-ADR / CAR-AGA numbering
    adr = by_reg.get("CAR-ADR", [])
    report("PASS" if any(re.fullmatch(r"\d+\.\d+", a["article_no"]) for a in adr) else "WARN", "CAR-ADR sections like 4.7")
    aga = by_reg.get("CAR-AGA-PART-ADR", [])
    report("PASS" if any(a["article_no"].startswith("ADR.OR.") for a in aga) else "WARN", "CAR-AGA rules like ADR.OR.C.020")
    report("PASS" if any(a["article_no"].startswith(("AMC", "GM")) for a in aga) else "WARN", "CAR-AGA AMC/GM entries")

    # Noise
    toc = [a["id"] for a in arts if re.search(r"(\.\s*){5,}|_{5,}", a["title"] + a["text"][:300])]
    report("PASS" if not toc else "WARN", "no table-of-contents leaders", f"{len(toc)} e.g. {toc[:3]}")
    hdr = [a["id"] for a in arts if re.search(r"Page\s*\d+\s*of\s*\d+", a["text"], re.I)]
    report("PASS" if not hdr else "WARN", "no running 'Page x of y' text", f"{len(hdr)} e.g. {hdr[:3]}")
    dups = [a["id"] for a in arts if re.search(r"-\d+$", a["article_no"])]
    report("PASS" if len(dups) < 20 else "WARN", "few '-2' duplicate numbers", f"{len(dups)} e.g. {dups[:5]}")

    print("\n10 longest articles (a very long one usually means a missed header pattern):")
    for a in sorted(arts, key=lambda a: -len(a["text"]))[:10]:
        print(f"  {len(a['text']):>7}  {a['id']}  {a['title'][:50]}")

    print("\n5 random CAR-DEF terms to check against the PDF:")
    random.seed(1)
    for a in random.sample(gl, min(5, len(gl))):
        print(f"  p{a['page']}  {a['title']}  ->  {a['text'][:100]}")

    fails = results.count("FAIL")
    print(f"\n{results.count('PASS')} pass · {results.count('WARN')} warn · {fails} fail")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "data/articles.json"))
