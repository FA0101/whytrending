# etl/fetch_x.py
import os
import time
import requests
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

X_BEARER = os.getenv("X_BEARER")

API = "https://api.twitter.com/2/tweets/search/recent"
# X API expects RFC3339 times (UTC) for start_time/end_time
# We’ll pass start_time for “last N hours”
TWEET_FIELDS = "created_at,lang,public_metrics"

def _headers():
    if not X_BEARER:
        raise RuntimeError("X_BEARER is not set in environment.")
    return {"Authorization": f"Bearer {X_BEARER}"}

def _rfc3339_hours_ago(hours: int) -> str:
    t = datetime.now(timezone.utc) - timedelta(hours=hours)
    return t.isoformat().replace("+00:00", "Z")

def search_twitter(query: str, max_results: int = 200, hours: int = 72) -> List[Dict]:
    """
    Twitter recent search (last ~7 days). Returns normalized rows with full metrics.
    Applies a start_time so results respect your window on the API side.
    """
    out: List[Dict] = []
    params = {
        "query": query,
        "max_results": 100,               # API cap per page (10..100)
        "tweet.fields": TWEET_FIELDS,
        "start_time": _rfc3339_hours_ago(max(1, hours)),  # at least 1 hour back
    }
    next_token: Optional[str] = None
    tries = 0

    while len(out) < max_results and tries < 12:
        if next_token:
            params["next_token"] = next_token
        try:
            r = requests.get(API, headers=_headers(), params=params, timeout=30)
            if r.status_code != 200:
                # Rate limit / auth hiccup — back off lightly
                time.sleep(2)
                tries += 1
                continue
            js = r.json()
            data = js.get("data", [])
            meta = js.get("meta", {}) or {}
            for t in data:
                pm = t.get("public_metrics", {}) or {}
                out.append({
                    "platform": "twitter",
                    "platform_id": t.get("id"),
                    "entity": None,  # pipeline fills this when looping entities
                    "text": t.get("text", ""),
                    "url": f"https://x.com/i/web/status/{t.get('id')}",
                    "created_at": t.get("created_at"),
                    "like_count": int(pm.get("like_count", 0)),
                    "comment_count": int(pm.get("reply_count", 0)),  # map reply->comment
                    "retweet_count": int(pm.get("retweet_count", 0)),
                    "reply_count": int(pm.get("reply_count", 0)),
                    "quote_count": int(pm.get("quote_count", 0)),
                    "lang": t.get("lang"),
                })
            next_token = meta.get("next_token")
            if not next_token:
                break
        except Exception:
            time.sleep(2)
            tries += 1

    return out[:max_results]
