import argparse, sqlite3, os, math, pandas as pd, numpy as np
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

DB_PATH = os.getenv("WHY_DB_PATH", "why_trending.db")

def load_posts(entity, start_ts, end_ts):
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """SELECT * FROM posts WHERE entity=? AND created_at BETWEEN ? AND ? ORDER BY created_at""" ,
        con, params=[entity, start_ts, end_ts]
    )
    con.close()
    return df

def keyword_delta(current_texts, baseline_texts, topk=15):
    from collections import Counter
    def tokenize(s):
        return [w for w in s.lower().split() if w.isalpha() or any(c.isalnum() for c in w)]
    cur_counts = Counter([w for t in current_texts for w in tokenize(t)])
    base_counts = Counter([w for t in baseline_texts for w in tokenize(t)])
    terms = set(cur_counts)|set(base_counts)
    scores = []
    for t in terms:
        cur = cur_counts[t] + 1
        base = base_counts[t] + 1
        scores.append((t, math.log(cur/base)))
    scores.sort(key=lambda x: x[1], reverse=True)
    # filter trivial words
    blacklist = set(["the","and","to","a","of","in","on","for","is","are","this","that","with","i","you","amp"])
    scores = [(t,s) for t,s in scores if t not in blacklist and len(t)>2][:topk]
    return scores

def cluster_topics(texts, k=None, top_terms=6):
    if len(texts) < 6:
        return []
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    X = embedder.encode(texts, normalize_embeddings=True)
    n = len(texts)
    if k is None:
        k = max(2, min(6, int(math.sqrt(n))))
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    y = km.fit_predict(X)
    # label clusters via TF-IDF
    vect = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
    tf = vect.fit_transform(texts)
    terms = np.array(vect.get_feature_names_out())
    clusters = []
    for ci in range(k):
        idx = np.where(y==ci)[0]
        if len(idx)==0: continue
        centroid = tf[idx].mean(axis=0).A1
        top_idx = centroid.argsort()[::-1][:top_terms]
        label_terms = [t for t in terms[top_idx] if len(t)>2][:top_terms]
        clusters.append({
            "cluster_id": int(ci),
            "size": int(len(idx)),
            "label": ", ".join(label_terms),
            "example_idx": int(idx[0])
        })
    clusters.sort(key=lambda c: c["size"], reverse=True)
    return clusters, y

def engagement_score(row):
    return int(row.get("like_count") or 0) + int(row.get("comment_count") or 0)

def main(entity, hours, baseline_days):
    end = datetime.utcnow()
    start = end - timedelta(hours=hours)
    base_start = start - timedelta(days=baseline_days)
    # Load
    cur_df = load_posts(entity, start.isoformat(), end.isoformat())
    base_df = load_posts(entity, base_start.isoformat(), start.isoformat())
    if cur_df.empty:
        print("No current-window posts. Run scraper first or widen window."); return
    if base_df.empty:
        print("Baseline empty. Consider increasing --baseline_days."); 
    # Keyword deltas
    kw = keyword_delta(cur_df["text"].fillna("").tolist(),
                       base_df["text"].fillna("").tolist() if not base_df.empty else [],
                       topk=15)
    # Clusters
    texts = cur_df["text"].fillna("").tolist()
    clusters = []
    cluster_assign = None
    if len(texts) >= 6:
        clusters, cluster_assign = cluster_topics(texts)
    # Representative posts (top 3 by engagement)
    cur_df["eng"] = cur_df.apply(engagement_score, axis=1)
    reps = (cur_df.sort_values("eng", ascending=False)
                .head(3)[["platform","url","text","eng","created_at"]].to_dict(orient="records"))
    # Save report JSON
    out = {
        "entity": entity,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "baseline_days": baseline_days,
        "top_keywords": [{"term":t,"score":float(s)} for t,s in kw],
        "clusters": clusters,
        "representative_posts": reps
    }
    os.makedirs("out", exist_ok=True)
    with open(f"out/why_{entity.replace(' ','_')}.json","w",encoding="utf-8") as f:
        import json; json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote -> out/why_{entity.replace(' ','_')}.json")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True, help="Name from configs/entities.yaml")
    ap.add_argument("--hours", type=int, default=48)
    ap.add_argument("--baseline_days", type=int, default=7)
    args = ap.parse_args()
    main(args.entity, args.hours, args.baseline_days)
