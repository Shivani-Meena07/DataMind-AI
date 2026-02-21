# 🧠 DataMind AI - Intelligent Database Documentation Agent

> **"An AI system that replaces tribal knowledge in databases."**

DataMind AI is a production-grade intelligent agent that automatically generates comprehensive, human-readable user manuals for any relational database by analyzing schema metadata and actual data patterns.

## 🎯 What Makes This Different

Unlike basic schema extractors or ER diagram tools, DataMind AI thinks like:
- **A Data Analyst** - Understanding data patterns, distributions, and quality
- **A Business Analyst** - Inferring business meaning from technical structures
- **A Senior Database Architect** - Understanding relationships, dependencies, and design patterns

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DataMind AI Agent Architecture                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────┐    ┌──────────────────────────────────────────────────────┐   │
│  │   Database   │───▶│              INGESTION LAYER                         │   │
│  │  Connection  │    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │   │
│  │   Manager    │    │  │   Schema    │  │    Data     │  │ Relationship│   │   │
│  └──────────────┘    │  │   Scanner   │  │   Profiler  │  │  Analyzer   │   │   │
│                      │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘   │   │
│                      └─────────┼────────────────┼────────────────┼──────────┘   │
│                                │                │                │              │
│                                ▼                ▼                ▼              │
│                      ┌──────────────────────────────────────────────────────┐   │
│                      │              ANALYSIS LAYER                           │   │
│                      │  ┌─────────────────────────────────────────────────┐  │   │
│                      │  │           Schema Intelligence Store             │  │   │
│                      │  │  • Table metadata    • Column profiles          │  │   │
│                      │  │  • Relationships     • Data quality metrics     │  │   │
│                      │  │  • Cardinality maps  • Pattern detection        │  │   │
│                      │  └─────────────────────────────────────────────────┘  │   │
│                      └──────────────────────────┬───────────────────────────┘   │
│                                                 │                               │
│                                                 ▼                               │
│                      ┌──────────────────────────────────────────────────────┐   │
│                      │              INFERENCE LAYER                          │   │
│                      │  ┌─────────────────────────────────────────────────┐  │   │
│                      │  │        LLM Context Inference Engine             │  │   │
│                      │  │  • Business entity recognition                  │  │   │
│                      │  │  • Semantic relationship interpretation         │  │   │
│                      │  │  • Use-case inference                           │  │   │
│                      │  │  • KPI suggestion                               │  │   │
│                      │  └─────────────────────────────────────────────────┘  │   │
│                      └──────────────────────────┬───────────────────────────┘   │
│                                                 │                               │
│                                                 ▼                               │
│                      ┌──────────────────────────────────────────────────────┐   │
│                      │              OUTPUT LAYER                             │   │
│                      │  ┌─────────────────────────────────────────────────┐  │   │
│                      │  │         Documentation Generator                 │  │   │
│                      │  │  • Executive Summary    • Entity Descriptions   │  │   │
│                      │  │  • Data Quality Report  • Sample Queries        │  │   │
│                      │  │  • Relationship Maps    • Caveats & Warnings    │  │   │
│                      │  └─────────────────────────────────────────────────┘  │   │
│                      └──────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key"

# Run with SQLite database
python -m datamind.main --db-type sqlite --db-path ./data/olist.db --output ./output/

# Run with PostgreSQL
python -m datamind.main --db-type postgresql --host localhost --port 5432 --database olist --user admin --password secret

# Run with MySQL
python -m datamind.main --db-type mysql --host localhost --port 3306 --database olist --user root --password secret
```

## 📦 Project Structure

```
datamind/
├── __init__.py
├── main.py                    # CLI and orchestration
├── config.py                  # Configuration management
├── core/
│   ├── __init__.py
│   ├── connection.py          # Database connection manager
│   ├── schema_scanner.py      # Metadata extraction
│   ├── data_profiler.py       # Data quality analysis
│   ├── relationship_analyzer.py # FK and relationship detection
│   └── intelligence_store.py  # Central data structure
├── inference/
│   ├── __init__.py
│   ├── llm_engine.py          # LLM integration
│   ├── prompts.py             # Prompt templates
│   └── entity_classifier.py   # Business entity classification
├── generators/
│   ├── __init__.py
│   ├── doc_generator.py       # Main documentation generator
│   ├── templates/             # Output templates
│   └── formatters.py          # Output formatting
├── utils/
│   ├── __init__.py
│   ├── sql_helpers.py         # SQL utilities
│   └── validators.py          # Data validation
└── demo/
    ├── setup_olist.py         # Demo database setup
    └── sample_queries.py      # Sample analytics queries
```

## 💡 Value Proposition

### For New Engineers
- **Zero ramp-up time**: Understand any database in minutes, not weeks
- **No tribal knowledge dependency**: Self-documenting databases
- **Instant context**: Business meaning behind every table and column

### For Organizations
- **Reduce onboarding costs**: New hires productive from day one
- **Eliminate documentation debt**: Always up-to-date docs
- **Compliance ready**: Audit-friendly documentation

### Scalability
- Handles 100+ table databases efficiently
- Parallel processing for large schemas
- Incremental updates for changing databases
- Caching layer for repeated analyses

## 📊 Supported Databases

| Database   | Status | Features |
|------------|--------|----------|
| PostgreSQL | ✅ Full | All features including advanced constraints |
| MySQL      | ✅ Full | All features |
| SQLite     | ✅ Full | All features |
| SQL Server | 🔄 Planned | Coming soon |
| Oracle     | 🔄 Planned | Coming soon |

## 📄 License

MIT License - See LICENSE file for details.
