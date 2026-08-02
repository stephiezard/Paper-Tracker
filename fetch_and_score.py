import os, json, re, datetime, time, urllib.request, urllib.parse

ANTHROPIC_KEY = os.environ["ANTHROPIC_API_KEY"]
API_URL = "https://api.anthropic.com/v1/messages"

with open("config/profile.json") as f:
    PROFILE = json.load(f)


def anthropic_call(model, system, user, max_tokens=500):
    body = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }).encode()
    req = urllib.request.Request(
        API_URL, data=body,
        headers={
            "content-type": "application/json",
            "x-api-key": ANTHROPIC_KEY,
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
    return "".join(b["text"] for b in data["content"] if b["type"] == "text")


def clean_json(text):
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return json.loads(text)


def fetch_europepmc(query, days=21):
    since = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()
    q = f"({query}) AND FIRST_PDATE:[{since} TO {datetime.date.today().isoformat()}]"
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urllib.parse.urlencode(
        {"query": q, "format": "json", "pageSize": 20}
    )
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.loads(r.read())
    results = []
    for item in data.get("resultList", {}).get("result", []):
        doi = item.get("doi", "")
        results.append({
            "title": (item.get("title") or "").strip(),
            "abstract": item.get("abstractText", "") or "",
            "source": item.get("source", ""),
            "doi": doi,
            "date": item.get("firstPublicationDate", ""),
            "link": f"https://doi.org/{doi}" if doi else "https://europepmc.org",
        })
    return results


def gather_candidates():
    seen = {}
    for kw in PROFILE["keywords"]:
        try:
            for p in fetch_europepmc(kw):
                key = p["doi"] or p["title"]
                if key and key not in seen and p["abstract"]:
                    seen[key] = p
        except Exception as e:
            print("fetch error for", kw, "->", e)
        time.sleep(1)
    print(f"Total unique candidates: {len(seen)}")
    return list(seen.values())


def score_candidates(candidates):
    scored = []
    for p in candidates[:40]:
        prompt = f"""Research profile: {PROFILE['summary']}
Trusted people/labs: {', '.join(PROFILE['trusted_people'])}

Paper title: {p['title']}
Abstract: {p['abstract'][:1200]}

Score 0-100 how relevant this paper is to the research profile above.
Reply with ONLY a JSON object like: {{"score": 82, "reason": "one sentence"}}"""
        try:
            out = anthropic_call(
                "claude-haiku-4-5-20251001",
                "You are a precise research relevance scorer. Reply with strict JSON only, no markdown fences.",
                prompt, max_tokens=150,
            )
            j = clean_json(out)
            p["score"] = int(j.get("score", 0))
            scored.append(p)
        except Exception as e:
            print("score error:", e)
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:8]


def summarize(paper):
    prompt = f"""Research profile: {PROFILE['summary']}

Paper title: {paper['title']}
Abstract: {paper['abstract'][:1500]}

Return ONLY a JSON object with exactly these keys:
"summary": 3 plain-English sentences,
"why": 1 sentence on why this matters to the profile above, wrapping the key phrase in <b></b> tags,
"figure": one figure worth inspecting based on the abstract, or "—" if unclear,
"verdict": one of "read fully", "inspect figures", "save only", "ignore" """
    out = anthropic_call(
        "claude-sonnet-5",
        "You are a scientific research assistant. Reply with strict JSON only, no markdown fences.",
        prompt, max_tokens=500,
    )
    return clean_json(out)


VERDICT_CLASS = {
    "read fully": "v-read",
    "inspect figures": "v-inspect",
    "save only": "v-save",
    "ignore": "v-save",
}


def main():
    candidates = gather_candidates()
    top = score_candidates(candidates)
    cards = []
    for p in top:
        try:
            s = summarize(p)
        except Exception as e:
            print("summarize error:", e)
            continue
        cards.append({
            "title": p["title"],
            "meta": f"{p['source']} · {p['date']}",
            "match": p["score"],
            "summary": s.get("summary", ""),
            "why": s.get("why", ""),
            "figure": s.get("figure", "—"),
            "figureCaption": "Fetched automatically — no image preview in v1.",
            "verdict": s.get("verdict", "save only"),
            "verdictClass": VERDICT_CLASS.get(s.get("verdict", "save only"), "v-save"),
            "link": p["link"],
        })
    os.makedirs("docs", exist_ok=True)
    with open("docs/today.json", "w") as f:
        json.dump({"date": datetime.date.today().isoformat(), "papers": cards}, f, indent=2)
    print(f"Wrote {len(cards)} cards to docs/today.json")


if __name__ == "__main__":
    main()
