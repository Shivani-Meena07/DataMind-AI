# 🧠 DataMind AI — Intelligent Database Documentation & Analytics Platform

> **"An AI system that replaces tribal knowledge in databases — No API Key Required."**

![DataMind AI](https://img.shields.io/badge/DataMind-AI-blue?style=for-the-badge)
![No API Key](https://img.shields.io/badge/API%20Key-Not%20Required-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-yellow?style=for-the-badge)
![Flask](https://img.shields.io/badge/Flask-3.0-red?style=for-the-badge)
![Charts](https://img.shields.io/badge/Charts-21%20Types-purple?style=for-the-badge)

DataMind AI is a **production-grade, zero-API-key** intelligent agent that automatically generates comprehensive documentation, interactive visualizations, natural-language SQL queries, and AI-powered search for any dataset — all running **100% locally** on your machine.

Upload a file → Get instant schema analysis, 21 enterprise charts, NL-to-SQL Lab, RAG-powered search, ER diagrams, data quality reports, and exportable documentation.

---

## 🎯 What Makes This Different

| Feature | Traditional Tools | DataMind AI |
|---------|-------------------|-------------|
| **API Key** | Required (OpenAI, etc.) | ❌ Not needed — ever |
| **Schema Docs** | Manual or basic extraction | Auto-generated with business descriptions |
| **Charts** | 2-3 basic charts | 21 enterprise-grade visualizations |
| **SQL Help** | Write it yourself | NL2SQL — ask in English (or Hindi!) |
| **Search** | Ctrl+F | Hybrid RAG engine (TF-IDF + BM25 + Keyword) |
| **Versioning** | None | Content-addressable semantic IDs |
| **Privacy** | Data sent to cloud | 100% offline — data never leaves your machine |

DataMind AI thinks like:
- 🔬 **A Data Analyst** — Understanding data patterns, distributions, and quality
- 💼 **A Business Analyst** — Inferring business meaning from technical structures
- 🏗️ **A Database Architect** — Understanding relationships, dependencies, and design patterns

---

## 🚀 Quick Start

### Web Application (Recommended)

```bash
# Clone and install
cd webapp
pip install -r requirements.txt

# Launch
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

**Windows shortcuts:**
- Double-click `webapp/run.bat`
- Or run `webapp/run.ps1` in PowerShell

### CLI Mode (Enterprise)

```bash
# Install core dependencies
pip install -r requirements.txt

# Run with SQLite
python -m datamind.main --db-type sqlite --db-path ./data/sample.db --output ./output/

# Run with PostgreSQL
python -m datamind.main --db-type postgresql --host localhost --port 5432 --database mydb --user admin --password secret

# Run with MySQL
python -m datamind.main --db-type mysql --host localhost --port 3306 --database mydb --user root --password secret
```

---

## 📁 Supported File Types

| Format | Extensions | Details |
|--------|------------|---------|
| **SQLite** | `.db`, `.sqlite`, `.sqlite3` | Full database with relationships, direct read-only SQL Lab |
| **CSV** | `.csv`, `.tsv`, `.txt` | Auto-detect delimiter (comma/tab), pandas + pure-Python fallback |
| **Excel** | `.xlsx`, `.xls`, `.xlsm` | Multi-sheet workbooks — each sheet becomes a separate table |
| **JSON** | `.json` | Array of objects or nested objects → auto-flattened to tables |
| **SQL Dump** | `.sql` | SQL dump file support |

---

## ✨ Features at a Glance

### 🖥️ 11-Tab Interactive Dashboard

| Tab | What It Does |
|-----|-------------|
| **📊 Overview** | Executive summary — table count, columns, rows, relationships, quality score |
| **📈 Charts** | Gallery of 21 auto-generated enterprise visualizations |
| **🔍 AI Search** | Natural language search over your data using hybrid RAG engine |
| **📋 Tables** | Detailed schema — columns, data types, PKs, FKs, semantic types, descriptions |
| **🔗 Relationships** | FK mappings with type (1:1, 1:N, N:M), explicit vs inferred |
| **✅ Quality** | Data quality score (0-100), issues by severity, recommendations |
| **📝 Queries** | Auto-generated SQL queries (SELECT, JOIN, aggregate) |
| **🧪 SQL Lab** | Type English questions → get SQL + live results (NL2SQL engine) |
| **🗺️ ER Diagram** | Interactive Mermaid-based entity-relationship diagram |
| **👀 Data Preview** | Browse raw table data (up to 500 rows) with table selector |
| **📤 Export** | Download full documentation as Markdown |

---

## 📈 21 Enterprise Chart Types

All charts are generated at **200 DPI** with enterprise styling, gradient aesthetics, and DataMind AI watermark.

### Table Analysis
| # | Chart | Description |
|---|-------|-------------|
| 1 | **Table Size Distribution** | Gradient horizontal bar chart of row counts |
| 2 | **Table Type Distribution** | Donut chart with glow effect (fact/dimension/bridge/lookup) |
| 3 | **Row Count Treemap** | Area-proportional treemap visualization |
| 4 | **Data Density Heatmap** | Grid heatmap of null vs populated data per column |
| 5 | **Table DNA Barcode** | Unique barcode-style fingerprint per table |

### Column Analysis
| # | Chart | Description |
|---|-------|-------------|
| 6 | **Column Data Types** | Gradient bar chart of type distribution |
| 7 | **Null Value Heatmap** | Heat grid of null percentages across all columns |
| 8 | **Null Distribution** | Horizontal bar chart of null rates |
| 9 | **Column Fingerprint** | Bubble scatter plot of column characteristics |

### Quality Analysis
| # | Chart | Description |
|---|-------|-------------|
| 10 | **Quality Score Gauge** | Neon gauge meter (0-100 scale) |
| 11 | **Quality Issues by Severity** | Grouped bar chart (high/medium/low) |
| 12 | **Completeness Radar** | Spider/radar chart of table completeness |

### Relationship Analysis
| # | Chart | Description |
|---|-------|-------------|
| 13 | **Network Relationship Graph** | Force-directed node-link graph |
| 14 | **Relationship Matrix Heatmap** | Table × table connection heatmap |
| 15 | **Dependency Hierarchy Tree** | Tree diagram of table dependencies |
| 16 | **Data Flow Sankey** | Sankey/alluvial diagram of data flow |
| 17 | **Schema Constellation** | Force-directed star map of schema |
| 18 | **FK Coverage Ring** | Donut ring showing foreign key coverage |

### Advanced / Summary
| # | Chart | Description |
|---|-------|-------------|
| 19 | **Sunburst Hierarchy** | Nested rings: Table → Column → Type |
| 20 | **Statistical Distribution Violin** | Violin plot of data distributions |
| 21 | **Schema Overview Matrix** | 4-panel composite summary dashboard |

---

## 🧪 NL2SQL Engine — Ask Questions in Plain English

The **zero-dependency, rule-based** Natural Language to SQL engine translates English (and Hindi!) questions into safe, executable SQL queries.

### Capabilities

- **10 Intent Types**: `COUNT`, `AVG`, `SUM`, `MAX`, `MIN`, `LIST`, `GROUP BY`, `COMPARE`, `TREND`, `FILTER`
- **Automatic JOIN Discovery**: 1-hop, 2-hop, and implicit column name matching
- **Cross-Table Aggregation**: Finds numeric columns across related tables
- **Smart Column Matching**: Semantic type-aware selection (`amount > quantity > score > measurement > year`)
- **Time Grouping**: Year/month/week/day with `strftime()` auto-detection
- **Comparison Operators**: `>`, `<`, `>=`, `<=`, `=`, `!=` from natural language
- **Top-N / Limit**: "top 10", "first 5", "limit 20"
- **Hindi Support**: `kitne`, `kitni` → count queries
- **Context-Aware Suggestions**: Dynamic suggestion chips based on actual dataset columns

### Examples

| Question | Generated SQL |
|----------|--------------|
| "How many customers?" | `SELECT COUNT(*) FROM customers` |
| "Average order value" | `SELECT AVG(order_value) FROM orders` |
| "Top 10 products by revenue" | `SELECT product_name, SUM(price) FROM products GROUP BY product_name ORDER BY 2 DESC LIMIT 10` |
| "Show orders from 2023" | `SELECT * FROM orders WHERE strftime('%Y', order_date) = '2023'` |

### Safety Sandbox
- ✅ Read-only enforcement — `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `TRUNCATE`, `ATTACH`, `DETACH`, `PRAGMA` are **blocked**
- ✅ Max 500 rows per query
- ✅ 10-second timeout
- ✅ Single-statement only (no `;` chaining)

---

## 🔍 Hybrid RAG Search Engine

Search your dataset analysis using natural language — powered by a **multi-strategy retrieval engine** with no external APIs.

### Search Strategies

| Strategy | Weight | Method |
|----------|--------|--------|
| **TF-IDF** | 40% | Cosine similarity with uni/bi/trigram support (10K features) |
| **BM25** | 35% | Okapi BM25 keyword relevance (k1=1.5, b=0.75) |
| **Keyword** | 25% | Inverted index exact match |
| **Semantic** | Optional | Sentence-transformers (`all-MiniLM-L6-v2`) deep embeddings |

### Features
- **MMR Re-ranking**: Maximal Marginal Relevance (λ=0.7) for diverse results
- **7 Chunk Types**: DB overview, table overview, column detail, relationship, quality issue, sample query, cross-table insight
- **Importance Weighting**: DB overview (2.0×) → table (1.5×) → relationship (1.3×) → quality (1.2×) → column (1.0×)
- **Persistent Index**: TF-IDF matrix, BM25 index, and chunks saved per dataset
- **LRU Cache**: Up to 10 vector stores in memory with auto-eviction
- **Intent Detection**: Automatically routes to table_info, column_info, relationship, quality, query_help, or stats answers

---

## 🆔 Semantic ID System — Content-Addressable Datasets

Every dataset gets a **human-readable, deterministic ID** based on its content:

```
NETFLIXMOVIE2020A3F2     ← Netflix titles dataset
DEMOECOMMERCE6CC7        ← E-commerce demo database  
OLISTCUSTOMER8A37        ← Olist customer dataset
```

### Format: `{SOURCE}{TOPIC}{YEAR}{HASH4}`

| Component | Source | Example |
|-----------|--------|---------|
| **Source** | 60+ known companies (Netflix, Amazon, Uber, Olist, Spotify, Airbnb, etc.) or first filename word | `NETFLIX` |
| **Topic** | 100+ keywords mapped across 30+ categories (e-commerce, finance, entertainment, health, etc.) | `MOVIE` |
| **Year** | Extracted from filename or data content (date columns, year values) | `2020` |
| **Hash** | 4-char hex from SHA-256 content hash for collision avoidance | `A3F2` |

### Why This Matters
- **Idempotent**: Same data → same ID → skip reprocessing (like `git` for datasets)
- **Deterministic**: Column-order and row-order independent content hashing
- **Change-sensitive**: Even a single cell change produces a new ID
- **Human-readable**: You can tell what a dataset contains from its ID alone

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DataMind AI Architecture                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────┐    ┌──────────────────────────────────────────────────────┐   │
│  │  File Upload │───▶│              INGESTION LAYER                         │   │
│  │  SQLite/CSV/ │    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │   │
│  │  Excel/JSON  │    │  │   Schema    │  │    Data     │  │ Relationship│   │   │
│  └──────────────┘    │  │   Scanner   │  │   Profiler  │  │  Analyzer   │   │   │
│                      │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘   │   │
│                      └─────────┼────────────────┼────────────────┼──────────┘   │
│                                │                │                │              │
│                                ▼                ▼                ▼              │
│                      ┌──────────────────────────────────────────────────────┐   │
│                      │           ANALYSIS & INTELLIGENCE LAYER               │   │
│                      │  ┌───────────┐ ┌───────────┐ ┌────────────────────┐  │   │
│                      │  │ Semantic  │ │ Quality   │ │ Business Desc.     │  │   │
│                      │  │ Type Inf. │ │ Scoring   │ │ Generation         │  │   │
│                      │  └───────────┘ └───────────┘ └────────────────────┘  │   │
│                      │  ┌───────────┐ ┌───────────┐ ┌────────────────────┐  │   │
│                      │  │ Content   │ │ FK Pattern│ │ Cardinality        │  │   │
│                      │  │ Hashing   │ │ Detection │ │ Analysis           │  │   │
│                      │  └───────────┘ └───────────┘ └────────────────────┘  │   │
│                      └──────────────────────────┬───────────────────────────┘   │
│                                                 │                               │
│              ┌──────────────────────────────────┼──────────────────────┐        │
│              ▼                                  ▼                      ▼        │
│  ┌────────────────────┐  ┌──────────────────────────┐  ┌───────────────────┐   │
│  │   OUTPUT LAYER     │  │    INTERACTIVE LAYER      │  │   SEARCH LAYER    │   │
│  │ • 21 Charts (PNG)  │  │ • NL2SQL Engine           │  │ • Hybrid RAG      │   │
│  │ • Markdown Docs    │  │ • SQL Lab (safe sandbox)  │  │ • TF-IDF + BM25   │   │
│  │ • Quality Reports  │  │ • ER Diagram (Mermaid)    │  │ • MMR Re-ranking  │   │
│  │ • Sample Queries   │  │ • Data Preview            │  │ • Intent Detection│   │
│  └────────────────────┘  └──────────────────────────┘  └───────────────────┘   │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                     STORAGE & VERSIONING LAYER                            │   │
│  │  Semantic IDs • Content Hashing • Deduplication • Auto-Migration          │   │
│  │  Atomic Writes • Integrity Verification • LRU Cache                       │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

### Backend
| Technology | Purpose |
|-----------|---------|
| **Python 3.8+** | Core runtime |
| **Flask 3.0** | Web framework with 23+ API routes |
| **SQLite** | Built-in analysis engine + SQL Lab runtime |
| **SQLAlchemy** | Enterprise multi-DB support (PostgreSQL, MySQL, SQLite) |
| **pandas + NumPy** | Data processing and analysis |
| **matplotlib** | 21 chart types (Agg backend, thread-safe) |
| **scikit-learn** | TF-IDF vectorization, cosine similarity |
| **scipy** | Sparse matrix storage for RAG index |
| **sentence-transformers** | Optional deep learning embeddings |

### Frontend
| Technology | Purpose |
|-----------|---------|
| **Vanilla JavaScript** | Single-page application (SPA), zero build step |
| **Tailwind CSS** | Utility-first responsive styling |
| **Font Awesome 6.4** | Icon library |
| **Mermaid.js** | Interactive ER diagrams |
| **Google Fonts (Noto Sans)** | Typography |
| **Custom CSS Animations** | Float, fade-in, slide-up, scale-in effects |

---

## 📦 Project Structure

```
datamind-ai/
├── README.md                          # This file
├── requirements.txt                   # Core/CLI dependencies
│
├── webapp/                            # 🌐 Web Application
│   ├── app.py                         # Flask backend (23+ routes, ~2300 lines)
│   ├── requirements.txt               # Web dependencies
│   ├── run.bat                        # Windows launcher
│   ├── run.ps1                        # PowerShell launcher
│   ├── templates/
│   │   └── index.html                 # SPA frontend (~3300 lines)
│   ├── versioning/                    # Core modules
│   │   ├── nl2sql.py                  # NL→SQL engine + SQL Playground
│   │   ├── embeddings.py             # RAG engine (TF-IDF + BM25 + Keyword)
│   │   ├── charts_advanced.py        # 21 chart generators
│   │   ├── charts.py                 # Chart orchestration
│   │   ├── storage.py                # Dataset storage & deduplication
│   │   ├── versioning.py             # Version management & caching
│   │   └── fingerprint.py            # Semantic ID generation (SHA-256)
│   └── storage/                       # Data storage (auto-created)
│       ├── index.json                 # Central dataset registry
│       ├── datasets/<id>/             # Raw files + schema + metadata
│       └── outputs/<id>/              # Analysis + charts + vector index
│
├── datamind/                          # 📦 Core Library (CLI/Enterprise)
│   ├── main.py                        # CLI orchestration
│   ├── config.py                      # YAML configuration
│   ├── core/
│   │   ├── connection.py              # Multi-DB connection manager
│   │   ├── schema_scanner.py          # Metadata extraction
│   │   ├── data_profiler.py           # Statistical profiling
│   │   ├── relationship_analyzer.py   # FK & relationship detection
│   │   └── intelligence_store.py      # Central data model
│   ├── inference/
│   │   ├── llm_engine.py             # Optional LLM integration
│   │   └── prompts.py                # Prompt templates
│   ├── generators/
│   │   └── doc_generator.py           # Documentation generation
│   └── demo/
│       ├── setup_olist.py             # Demo database generator
│       └── run_demo.py                # Demo runner
│
└── docs/
    └── ENTERPRISE_GUIDE.md            # Enterprise deployment guide
```

---

## 🔌 API Reference (23 Endpoints)

### Upload & Analysis
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload file (auto-fingerprinting, dedup) |
| `POST` | `/api/demo` | Generate built-in e-commerce demo (9 tables) |
| `POST` | `/api/analyze` | Run full analysis with caching |

### Dataset Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/datasets` | List all datasets |
| `GET` | `/api/datasets/<id>` | Get dataset metadata |
| `GET` | `/api/datasets/<id>/analysis` | Get analysis results |
| `GET` | `/api/datasets/<id>/verify` | Verify data integrity |
| `DELETE` | `/api/datasets/<id>` | Delete dataset + outputs |

### Charts
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/datasets/<id>/generate-charts` | Generate all 21 charts |
| `GET` | `/api/datasets/<id>/charts` | List available charts |
| `GET` | `/api/datasets/<id>/charts/<name>` | Get chart image (PNG) |

### SQL Lab & Data
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/datasets/<id>/nl2sql` | NL → SQL translation + execution |
| `POST` | `/api/datasets/<id>/execute-sql` | Execute raw SQL (read-only sandbox) |
| `GET` | `/api/datasets/<id>/preview/<table>` | Preview table data |
| `GET` | `/api/datasets/<id>/tables-list` | List available tables |
| `GET` | `/api/datasets/<id>/er-diagram` | Mermaid ER diagram + JSON |

### Search & Export
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/search` | RAG natural language search |
| `POST` | `/api/datasets/<id>/index` | Build/rebuild RAG index |
| `GET` | `/api/datasets/<id>/index-stats` | Vector index statistics |
| `GET` | `/api/export/markdown` | Export as Markdown (JSON) |
| `GET` | `/api/download/markdown` | Download Markdown file |
| `GET` | `/api/datasets/<id>/markdown` | Get dataset Markdown |
| `GET` | `/api/versioning/stats` | Cache & versioning stats |

---

## 🔬 Analysis Engine

### Semantic Type Detection (17 Rules)
Automatically infers business meaning from column names and data patterns:

| Semantic Type | Pattern Examples |
|---------------|-----------------|
| `identifier` | `*_id`, `*_key`, `*_code`, `*_number` |
| `name` | `*_name`, `first_name`, `last_name`, `title` |
| `email` | `*email*`, contains `@` in samples |
| `phone` | `*phone*`, `*mobile*`, `*tel*` |
| `address` | `*address*`, `*street*`, `*city*`, `*zip*` |
| `date` | `*date*`, `*_at`, `*_time`, `created`, `updated` |
| `amount` | `*price*`, `*cost*`, `*amount*`, `*total*`, `*revenue*` |
| `quantity` | `*quantity*`, `*count*`, `*qty*`, `*num_*` |
| `percentage` | `*percent*`, `*ratio*`, `*rate*` |
| `status` | `*status*`, `*state*`, `*flag*`, `is_*`, `has_*` |
| `description` | `*description*`, `*comment*`, `*note*`, `*text*` |
| `url` | `*url*`, `*link*`, `*website*`, `*href*` |
| `category` | `*category*`, `*type*`, `*class*`, `*group*` |
| `score` | `*score*`, `*rating*`, `*rank*`, `*grade*` |
| `coordinate` | `*lat*`, `*lng*`, `*longitude*`, `*latitude*` |
| `year` | `*year*`, `release_year`, `start_year` |
| `measurement` | `*weight*`, `*height*`, `*length*`, `*size*` |

### Table Type Classification
- **Fact tables**: High FK count, event/transaction patterns
- **Dimension tables**: Low FK count, reference/lookup patterns
- **Bridge tables**: Many-to-many relationship bridges
- **Lookup tables**: Small tables with category/status data

### Relationship Detection (7 Algorithms)
1. Semantic name matching (N-gram Jaccard similarity)
2. Data type compatibility scoring (5 type families)
3. FK pattern recognition (8+ patterns: `{table}_id`, `fk_{table}`, etc.)
4. Cardinality analysis (1:1, 1:N, N:1, N:M)
5. Value distribution correlation
6. Column uniqueness scoring
7. Pattern-based inference

### Quality Scoring
- **0-100 scale** with severity-weighted deductions
- **High severity** (-10): >50% null columns, empty tables
- **Medium severity** (-5): Low cardinality identifiers
- **Low severity** (-2): Minor data quality issues
- Actionable recommendations for each issue

---

## 🔒 Privacy & Security

| Feature | Detail |
|---------|--------|
| **100% Offline** | No external API calls — all processing runs locally |
| **No API Keys** | Zero dependency on cloud AI services |
| **Read-Only SQL** | `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `TRUNCATE`, `ATTACH`, `DETACH`, `PRAGMA` all blocked |
| **SQLite Read-Only** | SQL Lab opens files with `?mode=ro` URI parameter |
| **Single Statement** | Multi-statement SQL (`;` chaining) rejected |
| **Row Limits** | Max 500 rows returned per query |
| **File Size Limit** | 100MB max upload |
| **Secure Filenames** | `werkzeug.secure_filename` for all uploads |
| **Atomic Writes** | Index written to temp file, then renamed (crash-safe) |
| **No Telemetry** | Zero analytics, tracking, or data collection |

---

## 💡 Value Proposition

### For New Engineers
- **Zero ramp-up time**: Understand any database in minutes, not weeks
- **No tribal knowledge dependency**: Self-documenting databases
- **Instant context**: Business meaning behind every table and column

### For Organizations
- **Reduce onboarding costs**: New hires productive from day one
- **Eliminate documentation debt**: Always up-to-date docs
- **Compliance ready**: Audit-friendly documentation with quality scores

### For Data Teams
- **NL2SQL**: Non-technical stakeholders can query data in plain English
- **21 chart types**: Instant visual understanding of any dataset
- **RAG Search**: Ask questions about your data, get instant answers

---

## 📊 Supported Databases (CLI Mode)

| Database | Status | Features |
|----------|--------|----------|
| SQLite | ✅ Full | All features + SQL Lab |
| PostgreSQL | ✅ Full | All features via SQLAlchemy |
| MySQL | ✅ Full | All features via SQLAlchemy |
| SQL Server | 🔄 Planned | Coming soon |
| Oracle | 🔄 Planned | Coming soon |

---

## 🏃 Demo Mode

Click **"Try Demo Database"** in the web UI to instantly generate a 9-table e-commerce database (Olist) with:
- 200+ orders, customers, products, sellers
- Cross-table relationships (FKs)
- Multiple data types (dates, amounts, categories, coordinates)
- Deterministic seed — same demo every time

---

## 💡 Tips

1. **Large files?** — Analysis may take longer for 100K+ row datasets
2. **Better results?** — Use descriptive column names (`customer_email` > `col_7`)
3. **Excel files?** — Each sheet becomes a separate table with relationships
4. **JSON files?** — Nested objects are auto-flattened to relational tables
5. **Same file?** — Re-uploading skips analysis — cached results returned instantly
6. **SQL Lab?** — Start with suggestion chips, then type custom questions

---

## 🤝 Contributing

Contributions welcome! Feel free to:
- 🐛 Report bugs
- 💡 Suggest features
- 🔧 Submit pull requests

---

## 📄 License

MIT License — Use freely for any purpose.

---

<p align="center">
  Made with ❤️ by <strong>DataMind AI Team</strong><br>
  <em>No API Key Required — Ever!</em>
</p>
