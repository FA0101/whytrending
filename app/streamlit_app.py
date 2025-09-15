import os
import json
import re
import sqlite3
import subprocess
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import Counter

import pandas as pd
import streamlit as st

# --- Load env (WHY_DB_PATH, tokens, etc.) ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

DB_PATH = os.getenv("WHY_DB_PATH", "why_trending.db")


# ---------------------- DB & IO Helpers ----------------------
def safe_connect(db_path: str):
    return sqlite3.connect(db_path)

def ensure_db_initialized() -> bool:
    try:
        con = safe_connect(DB_PATH)
        rows = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='posts'"
        ).fetchall()
        con.close()
        return len(rows) == 1
    except Exception:
        return False

@st.cache_data
def load_entities():
    import yaml
    try:
        with open("configs/entities.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return [p["name"] for p in (cfg.get("people") or [])]
    except FileNotFoundError:
        return []

def add_person_to_config(name: str):
    import yaml
    path = "configs/entities.yaml"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {"people": []}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            existing = yaml.safe_load(f) or {}
            data = existing if isinstance(existing, dict) else {"people": []}
    people = data.get("people") or []
    if not any(p.get("name") == name for p in people):
        people.append({
            "name": name,
            "aliases": [name, f"#{name}"],
            "facebook_pages": [],
            "hashtags": [],
            "instagram_users": [],
            "instagram_hashtags": []
        })
        data["people"] = people
        with open(path, "w", encoding="utf-8") as f:
            import yaml as _yaml
            _yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        return True
    return False

def run_scraper_async(hours: int) -> str:
    """
    Start the scraper in a detached subprocess and return the path to the log file.
    """
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())
    log_path = logs_dir / f"scrape_{stamp}.log"

    DETACHED_PROCESS = 0x00000008  # Windows

    lf = open(log_path, "w", encoding="utf-8", buffering=1)
    subprocess.Popen(
        ["python", "-m", "pipeline.run_scrape", "--hours", str(hours)],
        stdout=lf,
        stderr=lf,
        creationflags=DETACHED_PROCESS
    )
    return str(log_path)

def run_why_async(entity: str, hours: int, baseline_days: int) -> str:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())
    log_path = logs_dir / f"why_{entity.replace(' ','_')}_{stamp}.log"

    DETACHED_PROCESS = 0x00000008  # Windows

    lf = open(log_path, "w", encoding="utf-8", buffering=1)
    subprocess.Popen(
        ["python", "-m", "processing.why_trending",
         "--entity", entity, "--hours", str(hours), "--baseline_days", str(baseline_days)],
        stdout=lf,
        stderr=lf,
        creationflags=DETACHED_PROCESS
    )
    return str(log_path)

def tail_log(path: str, lines: int = 120) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = f.read().splitlines()
        return "\n".join(data[-lines:])
    except FileNotFoundError:
        return "(log not created yet)"

def load_report(entity: str):
    path = f"out/why_{entity.replace(' ','_')}.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ---------------------- Twitter-only Loaders ----------------------
@st.cache_data
def load_posts_for_entity(entity: str, since_hours: int = 72, languages=None) -> pd.DataFrame:
    """Twitter-only entity match."""
    try:
        con = safe_connect(DB_PATH)
        start = (datetime.utcnow() - timedelta(hours=since_hours)).isoformat()
        df = pd.read_sql_query(
            """
            SELECT * FROM posts
            WHERE platform='twitter' AND entity=? AND created_at >= ?
            ORDER BY created_at DESC
            """,
            con, params=[entity, start]
        )
        con.close()
        if df.empty:
            return df
        if languages and "lang" in df.columns and not (len(languages) == 1 and languages[0] == "any"):
            df = df[df["lang"].fillna("unknown").isin(languages)]
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_posts_by_text(name_query: str, since_hours: int = 72, languages=None) -> pd.DataFrame:
    """Twitter-only free-text search."""
    try:
        con = safe_connect(DB_PATH)
        start = (datetime.utcnow() - timedelta(hours=since_hours)).isoformat()
        df = pd.read_sql_query(
            """
            SELECT * FROM posts
            WHERE platform='twitter' AND text LIKE ? AND created_at >= ?
            ORDER BY (like_count + comment_count + COALESCE(retweet_count,0) + COALESCE(reply_count,0) + COALESCE(quote_count,0)) DESC
            """,
            con, params=[f"%{name_query}%", start]
        )
        con.close()
        if df.empty:
            return df
        if languages and "lang" in df.columns and not (len(languages) == 1 and languages[0] == "any"):
            df = df[df["lang"].fillna("unknown").isin(languages)]
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_twitter_by_keyword(keyword: str, since_hours: int = 72, limit: int = 500, languages=None) -> pd.DataFrame:
    """Twitter-only keyword search used by the Quick Keyword Summary."""
    try:
        con = safe_connect(DB_PATH)
        start = (datetime.utcnow() - timedelta(hours=since_hours)).isoformat()
        df = pd.read_sql_query(
            """
            SELECT created_at, platform, entity, text, url, like_count, comment_count,
                   COALESCE(retweet_count,0) AS retweet_count,
                   COALESCE(reply_count,0)   AS reply_count,
                   COALESCE(quote_count,0)   AS quote_count,
                   COALESCE(lang,'unknown')  AS lang,
                   COALESCE(sent_label,'neutral') AS sent_label
            FROM posts
            WHERE platform='twitter' AND text LIKE ? AND created_at >= ?
            ORDER BY (like_count + comment_count + retweet_count + reply_count + quote_count) DESC
            LIMIT ?
            """,
            con, params=[f"%{keyword}%", start, limit]
        )
        con.close()
        if df.empty:
            return df
        if languages and not (len(languages) == 1 and languages[0] == "any"):
            df = df[df["lang"].fillna("unknown").isin(languages)]
        return df
    except Exception:
        return pd.DataFrame()


# ---------------------- Summaries & Picks ----------------------
_WORD_RE = re.compile(r"[#@\w']+", re.UNICODE)
_STOPWORDS = {
    "the","a","an","and","or","but","if","so","to","of","in","on","for","with","at","by","from",
    "is","am","are","was","were","be","been","being","this","that","these","those","it","its","as",
    "i","you","he","she","we","they","them","me","my","your","his","her","our","their","rt","via",
    "http","https","co","t","com","www","—","–"
}

def _tokenize(text: str):
    for w in _WORD_RE.findall(text.lower()):
        if len(w) < 3:
            continue
        if w in _STOPWORDS:
            continue
        yield w

def summarize_keyword_rows(df: pd.DataFrame, query: str) -> dict:
    n = len(df)
    if n == 0:
        return {
            "summary": f"No results for '{query}' in the selected window.",
            "top_terms": [],
            "sentiment": {},
            "examples": []
        }

    sent_counts = Counter((df["sent_label"].fillna("neutral")).astype(str).str.lower())
    total_eng = int(
        (df["like_count"].fillna(0) + df["comment_count"].fillna(0) +
         df.get("retweet_count", 0) + df.get("reply_count", 0) + df.get("quote_count", 0)).sum()
    )

    terms = Counter()
    for txt in df["text"].astype(str):
        terms.update(_tokenize(txt))
    top_terms = [w for w, _ in terms.most_common(12)]

    # top examples by engagement
    tmp = df.copy()
    tmp["eng"] = (tmp["like_count"].fillna(0) + tmp["comment_count"].fillna(0) +
                  tmp.get("retweet_count", 0) + tmp.get("reply_count", 0) + tmp.get("quote_count", 0))
    reps = tmp.sort_values("eng", ascending=False).head(3)[
        ["platform", "url", "text", "eng", "created_at"]
    ].to_dict("records")

    pos = sent_counts.get("positive", 0)
    neg = sent_counts.get("negative", 0)
    neu = sent_counts.get("neutral", 0)
    lead_terms = ", ".join(top_terms[:6]) if top_terms else "—"

    summary = (
        f"Found **{n}** tweets about **“{query}”** (total engagement ≈ **{total_eng}**). "
        f"Sentiment — positive: {pos}, neutral: {neu}, negative: {neg}. "
        f"Common terms: {lead_terms}."
    )
    return {"summary": summary, "top_terms": top_terms, "sentiment": dict(sent_counts), "examples": reps}

def pick_first_and_popular(df: pd.DataFrame):
    if df.empty:
        return None, None
    d = df.copy()
    d["eng"] = (d["like_count"].fillna(0) + d["comment_count"].fillna(0) +
                d.get("retweet_count", 0) + d.get("reply_count", 0) + d.get("quote_count", 0))
    try:
        d["_ts"] = pd.to_datetime(d["created_at"], errors="coerce", utc=True)
    except Exception:
        d["_ts"] = pd.NaT
    first_row = d.sort_values("_ts", ascending=True).iloc[0] if d["_ts"].notna().any() else d.tail(1).iloc[0]
    pop_row = d.sort_values("eng", ascending=False).iloc[0]
    return first_row, pop_row


# ---------------------- UI ----------------------
st.set_page_config(page_title="Why Trending (Twitter)", layout="wide")
st.title("🐦 Why Trending — Twitter/X")

# Guard: DB presence — add in-app initialize
if not ensure_db_initialized():
    st.warning("Database not initialized yet.")
    col_init, col_tip = st.columns([1, 3])
    with col_init:
        if st.button("Initialize database", key="btn_init_db"):
            try:
                import sys
                sys.path.insert(0, os.path.abspath("."))
                from db.db import init_db, migrate_db
                init_db(); migrate_db()
                st.success("Database initialized. You can now scrape.")
                st.rerun()
            except Exception as e:
                st.error(f"Init failed: {e}")
    with col_tip:
        st.caption("Alternatively, initialize via terminal: `python -m pipeline.run_scrape --hours 24` (it will also create the table).")
    st.stop()

entities = load_entities()

# ========================= Entity Flow (Twitter only) =========================
colA, colB, colC = st.columns([2, 1, 1])
with colA:
    st.subheader("Pick from watchlist **or** search any name")
    preset_choice = st.selectbox(
        "Watchlist",
        entities,
        index=0 if entities else None,
        placeholder="(none yet)",
        key="entity_select"
    )
    free_text = st.text_input(
        "Or enter any name (e.g., Burna Boy)",
        value="",
        help="Search in DB by text; or add to watchlist then scrape fresh.",
        key="entity_text"
    )
with colB:
    hours = st.slider(
        "Window (hours)",
        min_value=12, max_value=336, value=72, step=12,
        key="entity_hours"
    )
with colC:
    baseline_days = st.slider(
        "Baseline (days)",
        min_value=3, max_value=30, value=7, step=1,
        key="entity_baseline"
    )

# Twitter language filter (optional)
lang_choices = ["any", "en", "sw"]
languages = st.multiselect("Twitter language filter", lang_choices, default=["any"], key="lang_multi")

entity = free_text.strip() if free_text.strip() else (preset_choice or "").strip()
if not entity:
    st.info("Enter a name above or add someone to your watchlist to begin.")

st.divider()

# ---- Controls: Add / Scrape (async) / Why Analysis (async)
act1, act2, act3 = st.columns([1.3, 1.8, 2])
with act1:
    if st.button("➕ Add to watchlist", key="btn_add_watchlist"):
        if not free_text.strip():
            st.warning("Type a name in the text box first.")
        else:
            added = add_person_to_config(free_text.strip())
            if added:
                st.success(f"Added **{free_text.strip()}** to watchlist. You can scrape now.")
                load_entities.clear()
            else:
                st.info("That name is already on your watchlist.")

with act2:
    if "scrape_log_path" not in st.session_state:
        st.session_state["scrape_log_path"] = ""
    if st.button("🧹 Scrape now (watchlist, async)", key="btn_scrape_watchlist"):
        log_path = run_scraper_async(hours)
        st.session_state["scrape_log_path"] = log_path
        st.success(f"Scraper started in background.\n\nLog: `{log_path}`")
    if st.session_state.get("scrape_log_path"):
        with st.expander("📜 Scrape progress"):
            cR1, cR2 = st.columns([1, 3])
            with cR1:
                if st.button("🔄 Refresh log", key="btn_refresh_scrape_log"):
                    st.session_state["__dummy_refresh_scrape__"] = time.time()
            with cR2:
                st.code(tail_log(st.session_state["scrape_log_path"], lines=120), language="text")

with act3:
    if "why_log_path" not in st.session_state:
        st.session_state["why_log_path"] = ""
    if st.button("🧠 Run Why Analysis (async)", key="btn_run_why"):
        if not entity:
            st.warning("Pick a watchlist name or enter a name to analyze.")
        else:
            log_path = run_why_async(entity, hours, baseline_days)
            st.session_state["why_log_path"] = log_path
            st.success(f"Analysis started in background.\n\nLog: `{log_path}`")
    if st.session_state.get("why_log_path"):
        with st.expander("🧪 Analysis progress"):
            cW1, cW2 = st.columns([1, 3])
            with cW1:
                if st.button("🔄 Refresh analysis log", key="btn_refresh_why_log"):
                    st.session_state["__dummy_refresh_why__"] = time.time()
            with cW2:
                st.code(tail_log(st.session_state["why_log_path"], lines=120), language="text")

# ---------------------- Report / Results ----------------------
if entity:
    rep = load_report(entity)
    st.subheader(f"Why is **{entity}** trending?")

    if rep:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Top new keywords**")
            kw_df = pd.DataFrame(rep.get("top_keywords", []))
            if not kw_df.empty:
                st.dataframe(kw_df, width="stretch", hide_index=True, key="df_top_keywords")
            else:
                st.caption("No keywords found in this window.")
        with c2:
            st.markdown("**Top clusters (labels)**")
            cl_df = pd.DataFrame(rep.get("clusters", []))
            if not cl_df.empty:
                st.dataframe(cl_df, width="stretch", hide_index=True, key="df_clusters")
            else:
                st.caption("No clusters for this window.")
        with c3:
            st.markdown("**Representative posts**")
            reps = rep.get("representative_posts", [])
            if reps:
                for i, p in enumerate(reps):
                    st.write(f"- **[{p['platform']}]** [{p['url']}]({p['url']}) — {p.get('eng', 0)} reactions")
                    txt = p.get("text", "")
                    st.caption((txt[:180] + "…") if len(txt) > 180 else txt)
            else:
                st.caption("No examples available.")
    else:
        st.info("No analysis report found yet for this name. Use **Add to watchlist → Scrape now → Run Why Analysis**.")

    st.divider()
    st.subheader("Recent tweets")

    posts_df = load_posts_for_entity(entity, since_hours=hours, languages=languages)
    if posts_df.empty and free_text.strip():
        posts_df = load_posts_by_text(free_text.strip(), since_hours=hours, languages=languages)

    if posts_df.empty:
        st.warning("No tweets found for the selected window. Try increasing the hours or scraping again.")
    else:
        cols = ["created_at", "platform", "lang", "sent_label", "entity", "text", "url",
                "like_count", "comment_count", "retweet_count", "reply_count", "quote_count"]
        present = [c for c in cols if c in posts_df.columns]
        st.dataframe(
            posts_df[present],
            width="stretch", hide_index=True, key="df_recent_posts"
        )

st.divider()

# ========================= Quick Keyword Summary (Twitter only) =========================
st.header("🧭 Quick Keyword Summary (Twitter)")

kw_col1, kw_col2, kw_col3 = st.columns([2, 1, 1])
with kw_col1:
    kw_query = st.text_input(
        "Enter any keyword or phrase",
        value="",
        placeholder="e.g., Champions League, Burna Boy",
        key="kw_query_input"
    )
with kw_col2:
    kw_hours = st.slider(
        "Keyword Window (hours)",
        min_value=12, max_value=336, value=72, step=12,
        key="kw_hours"
    )
with kw_col3:
    if st.button("🔄 Add to watchlist & scrape now (async)", key="btn_kw_fetch"):
        q = kw_query.strip()
        if not q:
            st.warning("Type a keyword first.")
        else:
            added = add_person_to_config(q)
            if added:
                st.info(f"Added **{q}** to watchlist; starting scrape…")
                load_entities.clear()
            log_path = run_scraper_async(kw_hours)
            st.success(f"Scraper started. Log: `{log_path}`")

# language filter applies here too
st.caption("Language filter above applies to keyword summary (Twitter only).")

if st.button("Summarize keyword", key="btn_kw_summarize"):
    q = kw_query.strip()
    if not q:
        st.warning("Please enter a keyword or phrase.")
    else:
        with st.spinner(f"Searching tweets that mention “{q}”…"):
            dfk = load_twitter_by_keyword(q, since_hours=kw_hours, limit=500, languages=languages)

        if dfk.empty:
            st.info("No tweets found. Try increasing hours or scrape again.")
        else:
            # FIRST + MOST POPULAR
            first_row, pop_row = pick_first_and_popular(dfk)

            st.subheader("🏁 First tweet in window")
            if first_row is not None:
                st.write(f"[Open]({first_row['url']}) • {first_row.get('created_at')} • lang={first_row.get('lang')}")
                st.caption((first_row.get("text") or "")[:300] + ("…" if len(first_row.get("text",""))>300 else ""))
            else:
                st.caption("—")

            st.subheader("🏆 Most popular tweet")
            if pop_row is not None:
                st.write(
                    f"[Open]({pop_row['url']}) • {pop_row.get('created_at')} • "
                    f"❤ {int(pop_row.get('like_count',0))} | 💬 {int(pop_row.get('comment_count',0))} | "
                    f"🔁 {int(pop_row.get('retweet_count',0))} | 💬↩ {int(pop_row.get('reply_count',0))} | 🔗 {int(pop_row.get('quote_count',0))}"
                )
                st.caption((pop_row.get("text") or "")[:300] + ("…" if len(pop_row.get("text",""))>300 else ""))
            else:
                st.caption("—")

            st.subheader("📝 Why it’s trending (concise summary)")
            res = summarize_keyword_rows(dfk, q)
            st.markdown(res["summary"])

            s1, s2 = st.columns(2)
            with s1:
                st.markdown("**Sentiment breakdown**")
                sd = pd.DataFrame([res["sentiment"]]).T.reset_index()
                sd.columns = ["sentiment", "count"]
                st.dataframe(sd, hide_index=True, width="stretch", key="df_kw_sentiment")
            with s2:
                st.markdown("**Top terms**")
                tt = pd.DataFrame({"term": res["top_terms"]})
                st.dataframe(tt, hide_index=True, width="stretch", key="df_kw_terms")

            st.markdown("**Representative tweets**")
            for i, p in enumerate(res["examples"]):
                st.write(
                    f"- **[{p['platform']}]** [{p['url']}]({p['url']}) — {int(p.get('eng', 0))} reactions • {p.get('created_at')}"
                )
                txt = (p.get("text") or "").strip()
                if txt:
                    st.caption((txt[:220] + "…") if len(txt) > 220 else txt)

# Footer tip
st.caption("Tip: Add new names/keywords → run async scrape → then summarize or run analysis. Set X_BEARER in your .env.")