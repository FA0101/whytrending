# etl/fetch_reddit.py  (public JSON fallback, no OAuth needed)
import requests, time
from typing import List, Dict

def search_reddit(q: str, limit: int = 80) -> List[Dict]:
    out = []
    url = "https://www.reddit.com/search.json"
    headers = {"User-Agent": "why-trending/0.1 by demo"}
    params = {"q": q, "sort": "new", "limit": min(limit, 100)}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=30)
        if r.status_code != 200:
            return out
        for child in r.json().get("data", {}).get("children", []):
            d = child.get("data", {})
            out.append({
                "platform": "reddit",
                "platform_id": d.get("id"),
                "entity": None,
                "text": ((d.get("title") or "") + "\n" + (d.get("selftext") or "")).strip(),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(d.get("created_utc", 0))),
                "url": "https://www.reddit.com" + (d.get("permalink") or ""),
                "like_count": int(d.get("score") or 0),
                "comment_count": int(d.get("num_comments") or 0),
                "retweet_count": 0,
                "reply_count": 0,
                "quote_count": 0,
                "lang": None,
            })
    except Exception:
        return out
    return out
