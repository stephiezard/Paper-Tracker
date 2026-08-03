import os, json, re, datetime, time, urllib.request, urllib.parse

ANTHROPIC_KEY = os.environ["ANTHROPIC_API_KEY"]
API_URL = "https://api.anthropic.com/v1/messages"

with open("config/profile.json") as f:
    PROFILE = json.load(f)

SEEN_PATH = "docs/seen.json"
SEEN_RETENTION_DAYS = 45  # how long a paper stays "already shown" before it can resurface


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


def load_seen():
    try:
        with open(SEEN_PATH) as f:
            data = json.load(f)
    except Exception:
        return {}
    cutoff = (datetime.date.today() - datetime.timedelta(days=SEEN_RETENTION_DAYS)).isoformat()
    return {k: v for k, v in data.items() if v >= cutoff}


def save_seen(seen_dict):
    os.makedirs("docs", exist_ok=True)
    with open(SEEN_PATH, "w") as f:
        json.dump(seen_dict, f, indent=2)


def _europepmc_request(query, days):
    since = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()
    q = f"({query}) AND FIRST_PDATE:[{since} TO {datetime.date.today().isoformat()}]"
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urllib.parse.urlencode(
        {"query": q, "format": "json", "pageSize": 20}
    )
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.loads(r.read())
    return data.get("hitCount", 0), data.get("resultList", {}).get("result", [])


def _parse_result(item, from_trusted_author=False):
    doi = item.get("doi", "")
    title = (item.get("title") or "").strip()
    abstract = item.get("abstractText", "") or ""
    content = abstract if abstract else title
    return {
        "title": title,
        "abstract": content,
        "has_real_abstract": bool(abstract),
        "authors": item.get("authorString", "") or "",
        "source": item.get("source", ""),
        "doi": doi,
        "date": item.get("firstPublicationDate", ""),
        "link": f"https://doi.org/{doi}" if doi else "https://europepmc.org",
        "from_trusted_author": from_trusted_author,
    }


def fetch_by_keyword(query, days=21):
    hit_count, raw = _europepmc_request(query, days)
    print(f"  keyword '{query}' -> hitCount={hit_count}, returned={len(raw)}")
    return [_parse_result(item) for item in raw]


def fetch_by_trusted_author(name, days=60):
    # Trusted-network papers get a longer lookback since we want to catch
    # them even if they publish less frequently than a broad keyword search.
    surname = name.strip().split()[-1]
    query = f'AUTH:"{surname}"'
    hit_count, raw = _europepmc_request(query, days)
    print(f"  author '{name}' (searching '{surname}') -> hitCount={hit_count}, returned={len(raw)}")
    return [_parse_result(item, from_trusted_author=True) for item in raw]


def gather_candidates(seen_before):
    pool = {}

    for kw in PROFILE["keywords"]:
        try:
            for p in fetch_by_keyword(kw):
                key = p["doi"] or p["title"]
                if key and key not in seen_before and p["abstract"]:
                    pool.setdefault(key, p)
        except Exception as e:
            print("keyword fetch error for", kw, "->", e)
        time.sleep(1)

    for person in PROFILE["trusted_people"]:
        try:
            for p in fetch_by_trusted_author(person):
                key = p["doi"] or p["title"]
                if key and key not in seen_before and p["abstract"]:
                    if key in pool:
                        pool[key]["from_trusted_author"] = True
                    else:
                        pool[key] = p
        except Exception as e:
            print("author fetch error for", person, "->", e)
        time.sleep(1)

    print(f"Total unique new candidates (after excluding already-seen): {len(pool)}")
    return list(pool.values())


def score_candidates(candidates):
    scored = []
    for p in candidates[:40]:
        author_note = f"\nAuthors: {p['authors']}" if p["authors"] else ""
        prompt = f"""Research profile: {PROFILE['summary']}
Trusted people/labs (papers involving these people should be scored highly if at all topically relevant): {', '.join(PROFILE['trusted_people'])}

Paper title: {p['title']}{author_note}
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
            score = int(j.get("score", 0))
            # Deterministic network boost rather than relying solely on the
            # model to notice the author match.
            if p.get("from_trusted_author"):
                score = min(100, score + 15)
            p["score"] = score
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
    seen_before = load_seen()
    candidates = gather_candidates(seen_before)
    top = score_candidates(candidates)
    cards = []
    today_str = datetime.date.today().isoformat()

    for p in top:
        try:
            s = summarize(p)
        except Exception as e:
            print("summarize error:", e)
            continue
        ring_label = "trusted network" if p.get("from_trusted_author") else "auto"
        cards.append({
            "ring": ring_label,
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
        key = p["doi"] or p["title"]
        seen_before[key] = today_str

    os.makedirs("docs", exist_ok=True)
    with open("docs/today.json", "w") as f:
        json.dump({"date": today_str, "papers": cards}, f, indent=2)
    save_seen(seen_before)
    print(f"Wrote {len(cards)} cards to docs/today.json; seen.json now tracks {len(seen_before)} papers")


if __name__ == "__main__":
    main()
