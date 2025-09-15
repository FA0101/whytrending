# db/db.py
import os
import sqlite3
from typing import Iterable, Dict, Any

DB_PATH = os.getenv("WHY_DB_PATH", "why_trending.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS posts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  platform TEXT NOT NULL,              -- 'twitter', 'reddit', etc.
  platform_id TEXT,                    -- tweet id, reddit id, etc.
  entity TEXT,                         -- watchlist person/keyword (optional for ad-hoc)
  text TEXT,
  url TEXT UNIQUE,                     -- canonical URL; used for dedup
  created_at TEXT,                     -- ISO-8601 UTC
  like_count INTEGER DEFAULT 0,
  comment_count INTEGER DEFAULT 0,     -- replies or comments
  retweet_count INTEGER DEFAULT 0,     -- twitter only
  reply_count INTEGER DEFAULT 0,       -- twitter only
  quote_count INTEGER DEFAULT 0,       -- twitter only
  lang TEXT,                           -- ISO language code (twitter)
  sent_label TEXT                      -- 'positive'|'neutral'|'negative'
);
CREATE INDEX IF NOT EXISTS idx_posts_created_at ON posts(created_at);
CREATE INDEX IF NOT EXISTS idx_posts_entity ON posts(entity);
CREATE INDEX IF NOT EXISTS idx_posts_platform ON posts(platform);
"""

# Columns we want to make sure exist (safe migration)
MIGRATION_COLUMNS = [
    ("retweet_count", "INTEGER", "0"),
    ("reply_count",   "INTEGER", "0"),
    ("quote_count",   "INTEGER", "0"),
    ("lang",          "TEXT",    "NULL"),
    ("platform_id",   "TEXT",    "NULL"),
]

def get_conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    con = get_conn()
    cur = con.cursor()
    for stmt in SCHEMA_SQL.strip().split(";\n"):
        if stmt.strip():
            cur.execute(stmt)
    con.commit()
    con.close()

def migrate_db():
    """Idempotent: adds any missing columns without dropping data."""
    con = get_conn()
    cur = con.cursor()
    cur.execute("PRAGMA table_info(posts)")
    existing = {row[1] for row in cur.fetchall()}  # set of column names
    for col, ctype, default in MIGRATION_COLUMNS:
        if col not in existing:
            cur.execute(f"ALTER TABLE posts ADD COLUMN {col} {ctype}")
            if default != "NULL":
                cur.execute(f"UPDATE posts SET {col}={default} WHERE {col} IS NULL")
    con.commit()
    con.close()

def insert_posts(rows: Iterable[Dict[str, Any]]):
    """
    Insert many posts with dedup on url.
    Expected keys: platform, platform_id, entity, text, url, created_at,
                   like_count, comment_count, retweet_count, reply_count, quote_count, lang, sent_label
    """
    con = get_conn()
    cur = con.cursor()
    sql = """
    INSERT OR IGNORE INTO posts
    (platform, platform_id, entity, text, url, created_at,
     like_count, comment_count, retweet_count, reply_count, quote_count, lang, sent_label)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    data = []
    for r in rows:
        data.append((
            r.get("platform"),
            r.get("platform_id"),
            r.get("entity"),
            r.get("text"),
            r.get("url"),
            r.get("created_at"),
            int(r.get("like_count") or 0),
            int(r.get("comment_count") or 0),
            int(r.get("retweet_count") or 0),
            int(r.get("reply_count") or 0),
            int(r.get("quote_count") or 0),
            r.get("lang"),
            r.get("sent_label"),
        ))
    if data:
        cur.executemany(sql, data)
    con.commit()
    con.close()
