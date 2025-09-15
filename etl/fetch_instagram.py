import os, requests
from datetime import datetime, timezone, timedelta

IG_TOKEN = os.getenv("IG_ACCESS_TOKEN")
GRAPH_BASE = "https://graph.facebook.com/v19.0"

def _get(url, params):
    params = dict(params or {})
    params["access_token"] = IG_TOKEN
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return None
    return r.json()

def search_instagram_user_media(user_id, query=None, limit=50, hours=48):
    """Fetch recent media from an Instagram Business/Creator user you manage.
    Filters by a query string in the caption if provided.
    You must have a valid IG user token and permissions.
    """
    if not IG_TOKEN or not user_id:
        return []
    url = f"{GRAPH_BASE}/{user_id}/media"
    fields = "id,caption,timestamp,permalink,like_count,comments_count"
    data = _get(url, {"fields": fields, "limit": limit})
    out = []
    if not data or "data" not in data: 
        return out
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    for item in data["data"]:
        ts = item.get("timestamp")
        try:
            dt = datetime.fromisoformat(ts.replace("Z","+00:00"))
        except Exception:
            dt = None
        if dt and dt < cutoff:
            continue
        caption = item.get("caption") or ""
        if query and query.lower() not in caption.lower():
            continue
        out.append({
            "platform": "instagram",
            "author": None,
            "text": caption,
            "created_at": ts,
            "url": item.get("permalink"),
            "like_count": int(item.get("like_count") or 0),
            "comment_count": int(item.get("comments_count") or 0)
        })
    return out

def search_instagram_hashtag(hashtag, limit=50, hours=48):
    """Fetch recent media for a hashtag. Requires app approval for Hashtag Search.
    Pass the hashtag without the leading '#'."""
    if not IG_TOKEN or not hashtag:
        return []
    # find hashtag id
    h = _get(f"{GRAPH_BASE}/ig_hashtag_search", {"q": hashtag})
    if not h or not h.get("data"):
        return []
    hid = h["data"][0]["id"]
    url = f"{GRAPH_BASE}/{hid}/recent_media"
    fields = "id,caption,timestamp,permalink,like_count,comments_count"
    data = _get(url, {"user_id": "me", "fields": fields, "limit": limit})
    out = []
    if not data or "data" not in data:
        return out
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    for item in data["data"]:
        ts = item.get("timestamp")
        try:
            dt = datetime.fromisoformat(ts.replace("Z","+00:00"))
        except Exception:
            dt = None
        if dt and dt < cutoff:
            continue
        caption = item.get("caption") or ""
        out.append({
            "platform": "instagram",
            "author": None,
            "text": caption,
            "created_at": ts,
            "url": item.get("permalink"),
            "like_count": int(item.get("like_count") or 0),
            "comment_count": int(item.get("comments_count") or 0)
        })
    return out
