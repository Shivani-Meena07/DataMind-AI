# 🧠 DataMind AI - Web Application

> **No API Key Required!** Generate beautiful documentation for any data file instantly.

![DataMind AI](https://img.shields.io/badge/DataMind-AI-blue?style=for-the-badge)
![No API Key](https://img.shields.io/badge/API%20Key-Not%20Required-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-yellow?style=for-the-badge)

---

## 📁 Supported File Types

| Format | Extensions | Description |
|--------|------------|-------------|
| **SQLite** | `.db`, `.sqlite`, `.sqlite3` | Full database with relationships |
| **CSV** | `.csv`, `.tsv`, `.txt` | Comma/tab-separated values |
| **Excel** | `.xlsx`, `.xls`, `.xlsm` | Multi-sheet workbooks |
| **JSON** | `.json` | Arrays or nested objects |

---

## ✨ Features

- **📊 Schema Analysis** - Automatically detect tables, columns, and data types
- **🔗 Relationship Mapping** - Discover foreign keys and implicit relationships  
- **📈 Data Profiling** - Analyze null rates, uniqueness, and distributions
- **🎯 Semantic Inference** - Understand business meaning without LLM APIs
- **📋 Quality Reports** - Identify data quality issues with recommendations
- **📝 Sample Queries** - Get useful SQL queries automatically generated
- **📄 Export Options** - Download as Markdown or JSON

---

## 🚀 Quick Start

### Option 1: Double-click (Windows)
```
Just double-click run.bat
```

### Option 2: PowerShell
```powershell
cd webapp
.\run.ps1
```

### Option 3: Python
```bash
cd webapp
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000** in your browser!

---

## 📖 How to Use

### 1️⃣ Upload Your Data
- Drag & drop any supported file:
  - **SQLite**: `.db`, `.sqlite`, `.sqlite3`
  - **CSV/TSV**: `.csv`, `.tsv`, `.txt`
  - **Excel**: `.xlsx`, `.xls`, `.xlsm`
  - **JSON**: `.json`
- Or click "Try Demo Database" to see it in action

### 2️⃣ Analyze
- Click "Analyze Data"
- Wait a few seconds for the magic to happen

### 3️⃣ Explore Results
- **Overview** - Data summary and table list
- **Tables** - Detailed schema with column descriptions
- **Relationships** - ER diagram and FK mappings
- **Quality** - Data quality score and issues
- **Queries** - Auto-generated SQL queries
- **Export** - Download your documentation

---

## 🎨 Screenshots

### Clean Upload Interface
```
┌─────────────────────────────────────────┐
│       Drop your data file here          │
│       or click to browse                │
│   SQLite • CSV • Excel • JSON           │
│  [Analyze Database]  [Try Demo]         │
└─────────────────────────────────────────┘
```

### Beautiful Results Dashboard
```
┌─────────────────────────────────────────┐
│  📊 9 Tables | 📋 47 Columns | 🗃️ 500 Rows │
├─────────────────────────────────────────┤
│  [Overview] [Tables] [Relationships]    │
│  [Quality] [Queries] [Export]           │
└─────────────────────────────────────────┘
```

---

## 🛠️ Technical Details

### Stack
- **Backend:** Flask 3.0
- **Frontend:** Vanilla JS, Modern CSS
- **Database:** SQLite analysis  
- **No External APIs:** 100% local processing

### How It Works (No LLM!)

Instead of using expensive LLM APIs, DataMind uses:

1. **Pattern Recognition** - Regex patterns to identify semantic types
2. **Naming Conventions** - Infer meaning from column/table names
3. **Data Analysis** - Statistical profiling for insights
4. **Rule Engine** - Business logic templates for descriptions

Example pattern matching:
```python
# Automatically detects email columns
if 'email' in column_name or '@' in sample_values:
    semantic_type = 'email'
    description = 'Email address for communication'
```

---

## 📁 Project Structure

```
webapp/
├── app.py              # Flask backend + analyzer
├── requirements.txt    # Python dependencies
├── run.bat            # Windows launcher
├── run.ps1            # PowerShell launcher
├── templates/
│   └── index.html     # Beautiful UI (single file)
└── README.md          # This file
```

---

## 🔒 Privacy

- **No data leaves your computer** - Everything runs locally
- **No API calls** - No external services used
- **No storage** - Files are processed in memory
- **No tracking** - Zero analytics or telemetry

---

## 📝 Supported File Types

✅ **Fully Supported:**
- SQLite databases (`.db`, `.sqlite`, `.sqlite3`)
- CSV files (`.csv`, `.tsv`, `.txt`)
- Excel workbooks (`.xlsx`, `.xls`, `.xlsm`) - multiple sheets
- JSON files (`.json`) - arrays or nested objects

🔜 **Coming Soon:**
- PostgreSQL (via connection string)
- MySQL (via connection string)
- Parquet files

---

## 💡 Tips

1. **Large files?** - Analysis may take longer, be patient
2. **Better results?** - Use descriptive column names
3. **Excel files?** - Each sheet becomes a separate table
4. **JSON files?** - Nested objects are flattened to tables
5. **Quality issues?** - Review recommendations in Quality tab

---

## 🤝 Contributing

Contributions welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

---

## 📄 License

MIT License - Use freely for any purpose.

---

<p align="center">
  Made with ❤️ by DataMind AI Team<br>
  <strong>No API Key Required - Ever!</strong>
</p>
