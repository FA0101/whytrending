CREATE TABLE IF NOT EXISTS posts(
    doc_id TEXT PRIMARY KEY,
    platform TEXT,
    entity TEXT,
    keyword TEXT,
    author TEXT,
    text TEXT,
    url TEXT,
    created_at TEXT,
    like_count INTEGER,
    comment_count INTEGER,
    sent_label TEXT,
    sent_score REAL
);

CREATE INDEX IF NOT EXISTS idx_posts_entity_created ON posts(entity, created_at);
CREATE INDEX IF NOT EXISTS idx_posts_platform ON posts(platform);
