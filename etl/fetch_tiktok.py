# Best-effort TikTok fetcher using tiktokapipy; prefer official Research API if you have access.
from tiktokapipy.async_api import AsyncTikTokAPI
import asyncio

async def _search_hashtag(hashtag, max_videos=30):
    out=[]
    async with AsyncTikTokAPI() as api:
        tag = await api.hashtag(name=hashtag.lstrip("#"))
        async for v in tag.videos:
            out.append({
              "platform":"tiktok",
              "author": getattr(v.author, "unique_id", None),
              "text": v.desc or "",
              "created_at": v.create_time.isoformat() if v.create_time else None,
              "url": f"https://www.tiktok.com/@{v.author.unique_id}/video/{v.id}",
              "like_count": int(v.stats.digg_count or 0),
              "comment_count": int(v.stats.comment_count or 0)
            })
            if len(out)>=max_videos: break
    return out

def search_tiktok_hashtag(tag, max_videos=30):
    try:
        return asyncio.run(_search_hashtag(tag, max_videos))
    except Exception:
        return []
