# pipeline/run_scrape.py
import os
import argparse
from typing import List, Dict

from dotenv import load_dotenv
load_dotenv()

from db.db import init_db, migrate_db, insert_posts
from etl.fetch_x import search_twitter
from nlp.sentiment import score_sentiment
import yaml

def load_entities_config():
    path = os.path.join("configs", "entities.yaml")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("people") or []

def generate_search_terms(people: List[Dict]) -> List[Dict]:
    out = []
    for p in people:
        name = p.get("name")
        aliases = p.get("aliases") or [name]
        out.append({"entity": name, "terms": list(dict.fromkeys([a for a in aliases if a]))})
    return out

def label_and_score(rows: List[Dict]) -> List[Dict]:
    for r in rows:
        label, _ = score_sentiment(r.get("text", ""))
        r["sent_label"] = label
    return rows

def dedup_by_url(rows: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for r in rows:
        u = r.get("url")
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(r)
    return out

def main(hours: int):
    init_db()
    migrate_db()

    people = load_entities_config()
    targets = generate_search_terms(people)

    all_rows: List[Dict] = []

    for tgt in targets:
        ent = tgt["entity"]
        for term in tgt["terms"]:
            try:
                tweets = search_twitter(term, max_results=200, hours=hours)
                for t in tweets:
                    t["entity"] = ent
                all_rows.extend(tweets)
            except Exception as e:
                print("[twitter error]", term, e)

    all_rows = label_and_score(all_rows)
    all_rows = dedup_by_url(all_rows)

    insert_posts(all_rows)
    print(f"Inserted {len(all_rows)} new posts (of {len(all_rows)} fetched).")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=72)
    args = ap.parse_args()
    main(args.hours)
