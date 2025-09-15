import os, requests
FB_TOKEN = os.getenv("FB_ACCESS_TOKEN")

def search_page_posts(page_id, query=None, limit=50):
    """Fetch recent posts from a public Page by ID/username; filter by query on client side."""
    if not FB_TOKEN:
        return []
    url = f"https://graph.facebook.com/v19.0/{page_id}/posts"
    params = {"fields":"message,created_time,permalink_url,comments.summary(true),likes.summary(true)",
              "limit":limit, "access_token":FB_TOKEN}
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return []
    out=[]
    for item in r.json().get("data",[]):
        txt = item.get("message","") or ""
        if query and query.lower() not in txt.lower():
            continue
        likes = item.get("likes",{}).get("summary",{}).get("total_count",0)
        comments = item.get("comments",{}).get("summary",{}).get("total_count",0)
        out.append({
          "platform":"facebook",
          "author": None,
          "text": txt,
          "created_at": item.get("created_time"),
          "url": item.get("permalink_url"),
          "like_count": likes,
          "comment_count": comments
        })
    return out
