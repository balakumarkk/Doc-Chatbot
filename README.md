# Doc Chatbot — RAG Pipeline

A modular pipeline that turns documentation websites into embedding-ready chunks for a RAG chatbot.

```
Web Pages  -->  [Scraper]  -->  clean_text/*.md  -->  [Chunker]  -->  chunks.jsonl  -->  [Embedder -> Vector DB -> LLM]
```

**Done:** Scraper + Chunker | **Next:** Embedder -> Vector DB -> Chat API

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Quick Start](#quick-start)
3. [Stage 1 — Scraper](#stage-1--scraper)
4. [Stage 2 — Chunker](#stage-2--chunker)
5. [Configuration Reference](#configuration-reference)
6. [Output File Formats](#output-file-formats)
7. [Dependencies](#dependencies)
8. [Roadmap](#roadmap)

---

## Project Structure

```
Doc Chatbot/
|
+-- main.py                  # Scraper CLI entry point
+-- chunk.py                 # Chunker CLI entry point
+-- scraper_config.yaml      # Unified config for scraper + chunker
+-- requirements.txt         # All Python dependencies
+-- urls.txt                 # (optional) URL list for the scraper
|
+-- scraper/                 # Scraper package
|   +-- __init__.py
|   +-- config.py            # YAML config loader + dataclasses
|   +-- fetcher.py           # HTTP fetch with UA rotation & retries
|   +-- extractor.py         # Trafilatura content extraction + post-cleaning
|   +-- crawler.py           # BFS link discovery (same-domain, depth-limited)
|   +-- storage.py           # File saving, slug generation, manifest.json
|   +-- logger.py            # Console + file logging setup
|
+-- chunker/                 # Chunker package
|   +-- __init__.py          # Public API: split_markdown, ChunkRecord, write_chunks_jsonl
|   +-- splitter.py          # Two-pass LangChain chunking pipeline
|   +-- storage.py           # Write ChunkRecord -> chunks.jsonl; URL lookup
|
+-- scraped_docs/            # All generated output (add to .gitignore)
    +-- clean_text/          # One .md file per scraped page
    +-- raw_html/            # Raw HTML snapshots (for debugging)
    +-- chunks.jsonl         # Chunker output -- one JSON line per chunk
    +-- manifest.json        # Scraper run log -- one entry per URL
    +-- scraper.log          # Full debug log
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Scrape documentation

```bash
# Scrape a fixed list of URLs
python main.py --url-file urls.txt

# Or crawl from a seed URL (follows links 1 level deep)
python main.py --seed https://docs.example.com/ --depth 1
```

### 3. Chunk the clean text

```bash
# Uses defaults from scraper_config.yaml
python chunk.py

# Overwrite existing chunks.jsonl
python chunk.py --overwrite

# Dry run: count chunks without writing anything
python chunk.py --dry-run --verbose
```

Output: `scraped_docs/chunks.jsonl` — one JSON object per chunk, ready to embed.

---

## Stage 1 — Scraper

### How It Works

```
URL(s)
  |
  v
fetcher.py   -- HTTP GET with User-Agent rotation, urllib3 retry logic (429/5xx),
  |              configurable rate limiting, robots.txt check, raw HTML saved to disk
  v
crawler.py   -- (crawl mode only) BFS link discovery: same-domain, depth-limited,
  |              path-prefix scoped, regex-pattern exclusions
  v
extractor.py -- trafilatura two-pass extraction (precision first, recall fallback)
  |              Post-cleans: strips nav/footer artifacts, collapses blank lines
  v
storage.py   -- Saves clean text as slug-named .md or .txt
                Updates manifest.json atomically (write-then-rename)
```

### Scraper CLI (`main.py`)

**Input mode — pick exactly one:**

| Flag | Description |
|------|-------------|
| `--urls URL [URL ...]` | Scrape specific URLs passed directly |
| `--url-file FILE` | Read URLs from a text file (one per line; `#` = comment) |
| `--seed URL` | Crawl from a seed URL up to `--depth` levels |

**Crawl options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--depth INT` | `1` | BFS crawl depth when using `--seed` |
| `--same-domain` / `--no-same-domain` | `true` | Restrict links to seed domain |

**Output options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir PATH` | `scraped_docs` | Root output directory |
| `--format md\|txt` | `md` | Output format for clean text |

**Fetch behaviour:**

| Flag | Default | Description |
|------|---------|-------------|
| `--delay SECS` | `1.0` | Seconds between requests (polite crawling) |
| `--timeout SECS` | `20` | HTTP request timeout |
| `--no-robots` | — | Skip robots.txt checks |
| `--no-raw` | — | Do not save raw HTML snapshots |
| `--no-resume` | — | Re-scrape URLs already in manifest |
| `--config FILE` | `scraper_config.yaml` | Path to YAML config |

**Usage examples:**

```bash
# Flat fetch, markdown output
python main.py --urls https://docs.openai.com/api https://docs.openai.com/guides

# From a URL file, 2-second delay
python main.py --url-file urls.txt --delay 2.0

# Crawl 2 levels deep from a seed
python main.py --seed https://docs.python.org/3/ --depth 2

# Force re-scrape (ignore manifest)
python main.py --url-file urls.txt --no-resume
```

### Scraper Module Reference

#### `scraper/config.py`
Loads `scraper_config.yaml` into a `ScraperConfig` dataclass tree.
Priority: **CLI args > YAML file > built-in defaults**.
Sections: `output`, `fetch`, `crawl`, `resume`, `logging`, `chunker`.

#### `scraper/fetcher.py`
- Rotates through 5 realistic desktop User-Agent strings
- `requests.Session` + `urllib3.Retry` with backoff on 429/5xx
- robots.txt checked and cached per domain (controlled by `respect_robots`)
- Saves raw HTML to `scraped_docs/raw_html/<slug>.html` when enabled
- Returns `FetchResult(url, html, status_code, ok, error)`

#### `scraper/extractor.py`
- **Pass 1 (precision):** `trafilatura.extract(favor_precision=True)` — avoids false positives
- **Pass 2 (recall):** Falls back to `favor_recall=True` if Pass 1 returns nothing
- Post-cleaning: removes nav/footer artifact lines (home, back, next, copyright...), collapses 3+ blank lines to 2
- Returns `ExtractResult(title, text, word_count, slug, date)`

#### `scraper/crawler.py`
- BFS (Breadth-First Search) from a seed URL up to `depth` levels deep
- Deduplicates via canonical URL (strips query string)
- `path_prefix_depth`: restricts crawl to first N path segments of seed URL
  - e.g. seed = `/AmazonS3/latest/userguide/X.html`, `path_prefix_depth=1` restricts to `/AmazonS3/*`
- `exclude_patterns`: list of regex strings — matching URLs are skipped
- `iterate_urls()`: flat fetch with no link following (used in URL-list mode)

#### `scraper/storage.py`
- `url_to_slug(url)`: URL -> filesystem-safe slug (e.g. `docs_openai_com_guides_agents`)
- `save_content(result, dir, fmt)`: writes `# Title\n\nbody` for `.md`, plain header for `.txt`
- `update_manifest(entry, path)`: atomic append/update to `manifest.json`
- `get_scraped_urls(path)`: returns set of already-scraped URLs for resume mode

#### `scraper/logger.py`
- Console handler: INFO level (configurable)
- File handler: `scraped_docs/scraper.log` at DEBUG level (configurable)

---

## Stage 2 — Chunker

### Chunking Strategy

Uses a **two-pass LangChain pipeline** for semantically coherent, token-bounded chunks.

```
clean_text/*.md
       |
       v
  Pass 1 -- MarkdownHeaderTextSplitter
       |    Splits document on # / ## / ### headings.
       |    Each section carries its heading as metadata:
       |    {"h1": "Quickstart", "h2": "Authentication"}
       v
  Pass 2 -- RecursiveCharacterTextSplitter (tiktoken-aware)
       |    Splits oversized sections into <=512-token chunks with 64-token overlap.
       |    Separator order: \n\n -> \n -> ". " -> " " -> ""
       v
  ChunkRecord list
       |    Fields: chunk_id, source_file, source_url,
       |            chunk_index, text, token_count, headings
       v
  chunks.jsonl (one JSON line per chunk)
```

**Key design decisions:**

| Decision | Rationale |
|----------|-----------|
| MarkdownHeaderTextSplitter first | Chunks stay within one section; heading context travels with the text |
| tiktoken `cl100k_base` tokeniser | Same tokeniser as OpenAI embedding models; good approximation for Anthropic models too |
| 512-token chunk size | Fits within all common embedding model context windows; balances recall and precision |
| 64-token overlap | Prevents information loss at chunk boundaries |
| 20-token minimum | Drops noisy single-line stubs that hurt retrieval quality |
| Deterministic `chunk_id` | First 16 hex chars of SHA-256(`source_file + chunk_index`) -- stable across re-runs, safe for vector DB upserts |

### Chunker CLI (`chunk.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir PATH` | `scraped_docs/clean_text` | Directory of `.md`/`.txt` files to chunk |
| `--output FILE` | `scraped_docs/chunks.jsonl` | Output JSONL file |
| `--chunk-size TOKENS` | `512` | Max tokens per chunk |
| `--overlap TOKENS` | `64` | Overlap tokens between consecutive chunks |
| `--min-tokens TOKENS` | `20` | Minimum tokens to keep a chunk (noise filter) |
| `--config FILE` | `scraper_config.yaml` | YAML config path |
| `--overwrite` | false | Overwrite output file instead of appending |
| `--dry-run` | false | Count chunks without writing any output |
| `--verbose` | false | Print per-file chunk counts to console |

**Usage examples:**

```bash
# Default run (reads all settings from scraper_config.yaml)
python chunk.py

# Smaller chunks for more granular retrieval
python chunk.py --chunk-size 256 --overlap 32 --overwrite

# Preview chunk counts without writing
python chunk.py --dry-run --verbose

# Custom paths
python chunk.py --input-dir my_docs/ --output my_chunks.jsonl --overwrite
```

### Chunker Module Reference

#### `chunker/splitter.py`
- `split_markdown(text, source_file, source_url, chunk_size, chunk_overlap, min_chunk_tokens)` — main public function
- `ChunkRecord` dataclass — one instance per output chunk
- `_count_tokens(text)` — tiktoken `cl100k_base`; gracefully falls back to `len/4` if tiktoken not installed
- `_make_chunk_id(source_file, index)` — first 16 hex chars of SHA-256, deterministic and stable

#### `chunker/storage.py`
- `write_chunks_jsonl(chunks, output_path, mode)` — streams ChunkRecords to JSONL; auto-creates parent dirs
- `url_for_file(relative_path, manifest_path)` — resolves original URL from `manifest.json` to populate `source_url`

---

## Configuration Reference

`scraper_config.yaml` is the single source of truth for both stages.
CLI flags always override YAML values.

```yaml
# ── Output ───────────────────────────────────────────────────────────────────
output:
  dir: "scraped_docs"          # Root output directory
  format: "md"                 # "md" or "txt" for clean text files
  save_raw_html: true          # Save raw HTML snapshots to raw_html/

# ── HTTP Fetching ─────────────────────────────────────────────────────────────
fetch:
  delay: 1.0                   # Seconds to wait between requests
  timeout: 20                  # HTTP request timeout in seconds
  retries: 3                   # Retries on 429 / 5xx errors
  respect_robots: false        # Honour robots.txt (set true to be safe)
  user_agent: ""               # Empty = rotate built-in User-Agent pool

# ── Crawl (used when depth > 0) ───────────────────────────────────────────────
crawl:
  depth: 0                     # 0 = flat fetch; 1+ = follow links N levels deep
  same_domain_only: true       # Do not follow links to other domains
  path_prefix_depth: 1         # Restrict crawl to first N path segments of seed
  exclude_patterns:            # Regex patterns; matching URLs are skipped
    - "/login"
    - "/search"
    - "\\.pdf$"

# ── Resume / Deduplication ───────────────────────────────────────────────────
resume:
  enabled: true                # Skip URLs already present in manifest.json

# ── Logging ───────────────────────────────────────────────────────────────────
logging:
  console_level: "INFO"        # DEBUG | INFO | WARNING | ERROR
  file_level: "DEBUG"          # Written to scraped_docs/scraper.log

# ── Chunker ───────────────────────────────────────────────────────────────────
chunker:
  input_dir: "scraped_docs/clean_text"
  output_file: "scraped_docs/chunks.jsonl"
  chunk_size: 512              # Max tokens per chunk
  chunk_overlap: 64            # Overlap tokens between chunks
  min_chunk_tokens: 20         # Discard chunks shorter than this
```

---

## Output File Formats

### `scraped_docs/clean_text/<slug>.md`

One file per scraped page. Named by URL slug (e.g. `developers_openai_com_api_docs_guides_agents.md`).

```markdown
# Page Title

Extracted body text, cleaned of navigation and footer artifacts...
```

### `scraped_docs/manifest.json`

JSON array. One entry per URL attempted. Used for resume deduplication.

```json
[
  {
    "url": "https://developers.openai.com/api/docs/guides/agents",
    "title": "Agents Guide | OpenAI API",
    "filepath": "clean_text/developers_openai_com_api_docs_guides_agents.md",
    "scrape_date": "2026-07-14T19:43:00+05:30",
    "word_count": 1482,
    "status": "success",
    "error": null
  }
]
```

`status` values: `"success"` | `"failed"`

### `scraped_docs/chunks.jsonl`

One JSON object per line. This is the primary input to the embedding stage.

```json
{
  "chunk_id":    "e1a83d4778354ac7",
  "source_file": "clean_text/developers_openai_com_api_docs_guides_agents.md",
  "source_url":  "https://developers.openai.com/api/docs/guides/agents",
  "chunk_index": 2,
  "text":        "## Tool Calling\nAgents can call tools by defining a list of tool schemas...",
  "token_count": 387,
  "headings":    {"h1": "Agents Guide", "h2": "Tool Calling"}
}
```

**Field reference:**

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | `str` | 16-char deterministic hex ID — use as vector DB upsert key |
| `source_file` | `str` | Relative path to source `.md` file |
| `source_url` | `str` | Original web URL (looked up from manifest; empty if not found) |
| `chunk_index` | `int` | 0-based position of this chunk within its source file |
| `text` | `str` | Clean chunk text, ready to embed |
| `token_count` | `int` | Approximate token count (cl100k_base tokeniser) |
| `headings` | `dict` | Section heading context: `{"h1": "...", "h2": "...", "h3": "..."}` |

---

## Dependencies

| Package | Version | Used by | Purpose |
|---------|---------|---------|---------|
| `trafilatura` | >=1.8 | Scraper | Main content extraction from HTML |
| `requests` | >=2.31 | Scraper | HTTP client |
| `beautifulsoup4` | >=4.12 | Scraper | Link extraction during BFS crawl |
| `lxml` | >=5.0 | Scraper | HTML parser backend for bs4 |
| `colorama` | >=0.4 | Scraper | Colored terminal log output |
| `pyyaml` | >=6.0 | Both | YAML config file parsing |
| `langchain-text-splitters` | >=0.3 | Chunker | MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter |
| `tiktoken` | >=0.7 | Chunker | Token counting with cl100k_base encoding |

```bash
pip install -r requirements.txt
```

---

## Roadmap

```
[DONE]  Stage 1 -- Scraper     ->  scraped_docs/clean_text/*.md
[DONE]  Stage 2 -- Chunker     ->  scraped_docs/chunks.jsonl
[NEXT]  Stage 3 -- Embedder    ->  embed each chunk (OpenAI / Cohere / local model)
[NEXT]  Stage 4 -- Vector DB   ->  upsert vectors + metadata (Pinecone / Chroma / Qdrant)
[NEXT]  Stage 5 -- Chat API    ->  FastAPI endpoint: query -> retrieve -> LLM -> answer
[NEXT]  Stage 6 -- Frontend    ->  Chat UI (React / Next.js)
```

### Stage 3 (Embedder) — design notes

`chunks.jsonl` is the direct input. Each line = one embedding vector.
Use `chunk_id` as the vector ID for upserts — idempotent and safe to re-run.

Recommended embedding models:

| Provider | Model | Dimensions | Notes |
|----------|-------|-----------|-------|
| OpenAI | `text-embedding-3-small` | 1536 | Best quality/cost ratio |
| OpenAI | `text-embedding-3-large` | 3072 | Highest accuracy |
| Cohere | `embed-english-v3.0` | 1024 | Good multilingual support |
| Local | `BAAI/bge-small-en-v1.5` | 384 | Runs on CPU, no API cost |
