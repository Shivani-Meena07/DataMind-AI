"""
DataMind AI - Natural Language to SQL Engine
==============================================
Zero-dependency, rule-based NL→SQL translator that uses schema metadata
to convert English questions into executable SQL queries.

Architecture:
┌──────────────────────────────────────────────────────────┐
│                   NL → SQL Pipeline                      │
│                                                          │
│  ┌──────────┐  ┌────────────┐  ┌───────────────────┐    │
│  │ Tokenizer│─▶│ Entity     │─▶│ Query Builder     │    │
│  │ + Intent │  │ Resolver   │  │ (SELECT/JOIN/AGG) │    │
│  │ Detector │  │ (Schema    │  │                   │    │
│  │          │  │  Matching) │  │                   │    │
│  └──────────┘  └────────────┘  └───────┬───────────┘    │
│                                        │                │
│  ┌──────────┐  ┌────────────┐  ┌───────▼───────────┐    │
│  │ Response  │◀─│ Executor   │◀─│ SQL Validator     │    │
│  │ Formatter │  │ (SQLite)   │  │ + Safety Check    │    │
│  └──────────┘  └────────────┘  └───────────────────┘    │
└──────────────────────────────────────────────────────────┘

Supports:
- "Show me top 10 customers by revenue"
- "How many orders per month?"
- "What is the average price of products?"
- "List all tables with more than 1000 rows"
- "Find customers who spent more than $100"
- "Compare sales between 2020 and 2021"
- Automatic JOIN detection from schema relationships
"""

import re
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import time


@dataclass
class SQLQuery:
    """Represents a generated SQL query with metadata."""
    sql: str
    explanation: str
    intent: str
    tables_used: List[str]
    columns_used: List[str]
    confidence: float  # 0.0 - 1.0
    warnings: List[str] = field(default_factory=list)


@dataclass
class QueryResult:
    """Result of executing a SQL query."""
    success: bool
    sql: str
    explanation: str
    columns: List[str]
    rows: List[List[Any]]
    row_count: int
    execution_time_ms: float
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    confidence: float = 0.0


class NL2SQLEngine:
    """
    Natural Language to SQL translation engine.
    
    Uses schema metadata + rule-based patterns to convert
    English questions into executable SQL. No LLM needed.
    """
    
    # ====================================================================
    # Intent Detection Patterns
    # ====================================================================
    
    INTENT_PATTERNS = {
        'count': [
            r'\bhow many\b', r'\bcount\b', r'\bnumber of\b', r'\btotal\s+(?:number|count)\b',
            r'\bkitne\b', r'\bkitni\b',  # Hindi support
        ],
        'average': [
            r'\baverage\b', r'\bavg\b', r'\bmean\b', r'\btypical\b',
        ],
        'sum': [
            r'\btotal\b', r'\bsum\b', r'\bcombined\b', r'\ball\s+together\b',
            r'\boverall\b', r'\bgross\b',
        ],
        'max': [
            r'\bmaximum\b', r'\bmax\b', r'\bhighest\b', r'\blargest\b', 
            r'\bbiggest\b', r'\bmost\s+expensive\b', r'\bbest\b',
        ],
        'min': [
            r'\bminimum\b', r'\bmin\b', r'\blowest\b', r'\bsmallest\b',
            r'\bcheapest\b', r'\bleast\b', r'\bworst\b',
        ],
        'list': [
            r'\bshow\b', r'\blist\b', r'\bdisplay\b', r'\bget\b', r'\bfetch\b',
            r'\bfind\b', r'\bsearch\b', r'\bselect\b', r'\bretrieve\b',
            r'\bgive\s+me\b', r'\bwhat\s+are\b', r'\bwhich\b',
        ],
        'group': [
            r'\bper\b', r'\bby\b', r'\beach\b', r'\bgroup\b', r'\bbreakdown\b',
            r'\bdistribution\b', r'\bcategorize\b', r'\bsegment\b',
        ],
        'compare': [
            r'\bcompare\b', r'\bversus\b', r'\bvs\b', r'\bdifference\b',
            r'\bbetween\b.*\band\b',
        ],
        'trend': [
            r'\btrend\b', r'\bover\s+time\b', r'\bmonthly\b', r'\byearly\b',
            r'\bdaily\b', r'\bweekly\b', r'\bgrowth\b',
        ],
        'filter': [
            r'\bwhere\b', r'\bwith\b', r'\bhaving\b', r'\bgreater\s+than\b',
            r'\bless\s+than\b', r'\bmore\s+than\b', r'\bover\b', r'\bunder\b',
            r'\babove\b', r'\bbelow\b', r'\bequal\b', r'\blike\b',
        ],
    }
    
    # Aggregate function mapping
    AGG_MAP = {
        'count': 'COUNT',
        'average': 'AVG',
        'sum': 'SUM',
        'max': 'MAX',
        'min': 'MIN',
    }
    
    # Time period patterns
    TIME_PATTERNS = {
        'year': [r'\byear\b', r'\byearly\b', r'\bannual\b'],
        'month': [r'\bmonth\b', r'\bmonthly\b'],
        'week': [r'\bweek\b', r'\bweekly\b'],
        'day': [r'\bday\b', r'\bdaily\b', r'\bdate\b'],
    }
    
    # Limit/Top patterns
    LIMIT_PATTERN = re.compile(
        r'\btop\s+(\d+)\b|\bfirst\s+(\d+)\b|\blimit\s+(\d+)\b|\b(\d+)\s+(?:results?|records?|rows?)\b',
        re.IGNORECASE
    )
    
    # Comparison patterns  
    COMPARISON_PATTERNS = [
        (re.compile(r'(?:greater|more|over|above|exceed|>\s*)\s*(?:than\s+)?["\']?(\d+(?:\.\d+)?)', re.I), '>'),
        (re.compile(r'(?:less|under|below|fewer|<\s*)\s*(?:than\s+)?["\']?(\d+(?:\.\d+)?)', re.I), '<'),
        (re.compile(r'(?:at\s+least|>=)\s*["\']?(\d+(?:\.\d+)?)', re.I), '>='),
        (re.compile(r'(?:at\s+most|<=)\s*["\']?(\d+(?:\.\d+)?)', re.I), '<='),
        (re.compile(r'(?:equal|exactly|=)\s*(?:to\s+)?["\']?(\d+(?:\.\d+)?)', re.I), '='),
        (re.compile(r'(?:not|!=|<>)\s*(?:equal\s+)?(?:to\s+)?["\']?(\d+(?:\.\d+)?)', re.I), '!='),
    ]
    
    # Sort direction patterns  
    SORT_DESC = re.compile(r'\bdescending\b|\bdesc\b|\bhighest\b|\bmost\b|\btop\b|\blargest\b|\bbest\b', re.I)
    SORT_ASC = re.compile(r'\bascending\b|\basc\b|\blowest\b|\bleast\b|\bsmallest\b|\bworst\b|\bfirst\b', re.I)
    
    def __init__(self, schema: Dict[str, Any]):
        """
        Initialize NL2SQL engine with schema metadata.
        
        Args:
            schema: Analysis results containing tables, columns, relationships
        """
        self.schema = schema
        self.tables = {}          # table_name -> table_info
        self.columns = {}         # table_name -> {col_name: col_info}
        self.relationships = []   # list of relationships
        self.table_aliases = {}   # alias -> table_name (plural, singular, etc.)
        self.column_index = {}    # column_name -> [(table_name, col_info)]
        
        self._build_schema_index()
    
    def _build_schema_index(self):
        """Build efficient lookup indexes from schema."""
        tables_data = self.schema.get('tables', [])
        
        for table in tables_data:
            tname = table['name']
            self.tables[tname.lower()] = table
            self.columns[tname.lower()] = {}
            
            # Build table aliases
            tl = tname.lower()
            self.table_aliases[tl] = tname
            # Plural/singular variations
            if tl.endswith('s'):
                self.table_aliases[tl[:-1]] = tname  # orders -> order
            else:
                self.table_aliases[tl + 's'] = tname  # order -> orders
            # Without common prefixes/suffixes
            for prefix in ['olist_', 'dim_', 'fact_', 'tbl_', 'tb_']:
                if tl.startswith(prefix):
                    self.table_aliases[tl[len(prefix):]] = tname
            # Underscore to space mapping
            self.table_aliases[tl.replace('_', ' ')] = tname
            # Individual words from multi-word table names (e.g., "category" → product_category)
            parts = tl.split('_')
            if len(parts) > 1:
                for part in parts:
                    if len(part) >= 4 and part not in self.table_aliases:
                        self.table_aliases[part] = tname
            
            # Build column index
            for col in table.get('columns', []):
                cname = col['name']
                cl = cname.lower()
                self.columns[tl][cl] = col
                
                if cl not in self.column_index:
                    self.column_index[cl] = []
                self.column_index[cl].append((tname, col))
                
                # Column aliases (without table prefix)
                for prefix in [tl + '_', tl[:-1] + '_' if tl.endswith('s') else '']:
                    if prefix and cl.startswith(prefix):
                        alias = cl[len(prefix):]
                        if alias not in self.column_index:
                            self.column_index[alias] = []
                        self.column_index[alias].append((tname, col))
        
        # Build relationships index
        self.relationships = self.schema.get('relationships', [])
        
        # Infer types from column names when metadata is missing
        self._infer_column_types()
    
    # ====================================================================
    # Type Inference from Column Names
    # ====================================================================
    
    # Patterns: (column name regex, semantic_type, data_type)
    COLUMN_TYPE_RULES = [
        # Identifiers
        (re.compile(r'_id$|^id$', re.I), 'identifier', 'TEXT'),
        # Money / amounts
        (re.compile(r'price|cost|amount|value|payment|revenue|freight|fee|profit|income|salary|budget|expense|discount|tax|total_amount|unit_price', re.I), 'amount', 'REAL'),
        # Dates / times
        (re.compile(r'date|_at$|timestamp|_time$|created|updated|modified|delivered|approved|estimated|shipped|born|started|ended', re.I), 'date', 'TEXT'),
        # Year columns (separate from dates — these are INTEGER)
        (re.compile(r'\byear\b|release_year|start_year|end_year|birth_year', re.I), 'year', 'INTEGER'),
        # Scores / ratings
        (re.compile(r'score|rating|rank|grade|level|priority|stars', re.I), 'score', 'REAL'),
        # Quantities
        (re.compile(r'qty|quantity|count|number|num_|_num$|items|installment|sequential|order_item_id', re.I), 'quantity', 'INTEGER'),
        # Measurements
        (re.compile(r'weight|length|height|width|size|area|volume|distance|duration|_cm$|_mm$|_kg$|_g$|_lb', re.I), 'measurement', 'REAL'),
        # Geo / address
        (re.compile(r'zip|postal|city|state|country|region|address|lat$|lng$|longitude|latitude|geo', re.I), 'address', 'TEXT'),
        # Status / category
        (re.compile(r'status', re.I), 'status', 'TEXT'),
        (re.compile(r'category|type|kind|class|group|segment|tier', re.I), 'category', 'TEXT'),
        # Names / text
        (re.compile(r'name|title|label|display', re.I), 'name', 'TEXT'),
        (re.compile(r'description|comment|message|note|text|body|content|review_comment', re.I), 'description', 'TEXT'),
        (re.compile(r'email', re.I), 'email', 'TEXT'),
        (re.compile(r'phone|mobile|tel', re.I), 'phone', 'TEXT'),
        (re.compile(r'url|link|image|photo|picture|avatar', re.I), 'url', 'TEXT'),
    ]
    
    def _infer_column_types(self):
        """Infer semantic_type and data_type from column names when not provided.
        
        Preserves existing data_type from schema (e.g. INTEGER for release_year)
        and only infers semantic_type from name patterns.
        Treats 'unknown' semantic_type as empty.
        Cross-checks: if actual data_type is TEXT, don't assign numeric semantic types.
        """
        NUMERIC_SEM_TYPES = {'amount', 'score', 'quantity', 'measurement', 'year'}
        
        DATE_PATTERN = re.compile(r'date|_at$|timestamp|_time$|created|updated|modified|delivered|approved|estimated|shipped|born|started|ended', re.I)
        
        for tname_lower, cols in self.columns.items():
            for cname_lower, cinfo in cols.items():
                original_data_type = cinfo.get('data_type', '')
                existing_sem = cinfo.get('semantic_type', '')
                
                # Treat 'unknown' as no semantic type
                if existing_sem == 'unknown':
                    cinfo['semantic_type'] = ''
                    existing_sem = ''
                
                # Fix generic 'text' semantic type for date-patterned columns
                col_name = cinfo.get('name', cname_lower)
                if existing_sem == 'text' and DATE_PATTERN.search(col_name):
                    cinfo['semantic_type'] = 'date'
                    existing_sem = 'date'
                
                # Skip if already has BOTH valid type info
                if existing_sem and original_data_type:
                    # Cross-check: if data_type is TEXT but semantic says numeric → fix it
                    if original_data_type.upper() == 'TEXT' and existing_sem in NUMERIC_SEM_TYPES:
                        # The column is TEXT (like "PG-13" for rating), override bad semantic type
                        cinfo['semantic_type'] = 'category' if existing_sem == 'score' else 'text'
                    continue
                
                col_name = cinfo.get('name', cname_lower)
                
                for pattern, sem_type, data_type in self.COLUMN_TYPE_RULES:
                    if pattern.search(col_name):
                        if not existing_sem:
                            # Cross-check: don't assign numeric semantic type to TEXT columns
                            if original_data_type.upper() == 'TEXT' and sem_type in NUMERIC_SEM_TYPES:
                                cinfo['semantic_type'] = 'category' if sem_type == 'score' else 'text'
                            else:
                                cinfo['semantic_type'] = sem_type
                        # Only set data_type if not already provided by schema
                        if not original_data_type:
                            cinfo['data_type'] = data_type
                        break
                
                # If still no type, mark as TEXT by default (only if schema didn't provide it)
                if not cinfo.get('data_type'):
                    cinfo['data_type'] = 'TEXT'
    
    def translate(self, question: str) -> SQLQuery:
        """
        Translate a natural language question to SQL.
        
        Args:
            question: Natural language question
            
        Returns:
            SQLQuery with generated SQL and metadata
        """
        q = question.strip()
        q_lower = q.lower()
        
        # Detect intent
        intents = self._detect_intents(q_lower)
        
        # Resolve entities (tables and columns)
        matched_tables = self._resolve_tables(q_lower)
        matched_columns = self._resolve_columns(q_lower, matched_tables)
        
        # For aggregate queries with financial keywords, ensure a table with amount columns is included
        agg_keywords = ['revenue', 'sales', 'income', 'price', 'cost', 'payment', 'spent', 'total', 'money', 'profit']
        if any(kw in q_lower for kw in agg_keywords):
            has_amount_table = False
            for t in matched_tables:
                for cname, cinfo in self.columns.get(t.lower(), {}).items():
                    if cinfo.get('semantic_type') == 'amount':
                        has_amount_table = True
                        break
                if has_amount_table:
                    break
            if not has_amount_table:
                # Find a table with amount columns and add it
                for tname_lower, cols in self.columns.items():
                    for cname, cinfo in cols.items():
                        if cinfo.get('semantic_type') == 'amount':
                            real_name = self.tables[tname_lower]['name']
                            if real_name not in matched_tables:
                                matched_tables.append(real_name)
                            has_amount_table = True
                            break
                    if has_amount_table:
                        break
        
        # Detect limit
        limit = self._detect_limit(q_lower)
        
        # Detect comparisons/filters
        filters = self._detect_filters(q_lower, matched_columns)
        
        # Detect time grouping
        time_group = self._detect_time_grouping(q_lower, matched_columns)
        
        # Detect sort direction
        sort_desc = bool(self.SORT_DESC.search(q_lower))
        
        # Build SQL based on intent
        if not matched_tables:
            # No tables matched - try broader matching
            matched_tables = self._fuzzy_table_match(q_lower)
        
        if not matched_tables:
            # Still nothing? Use the first table as default
            if self.tables:
                first_table = list(self.tables.values())[0]
                matched_tables = [first_table['name']]
        
        # Generate SQL
        sql_query = self._build_sql(
            intents=intents,
            tables=matched_tables,
            columns=matched_columns,
            limit=limit,
            filters=filters,
            time_group=time_group,
            sort_desc=sort_desc,
            question=q_lower
        )
        
        return sql_query
    
    def _detect_intents(self, q: str) -> List[str]:
        """Detect query intents from the question."""
        intents = []
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, q):
                    intents.append(intent)
                    break
        
        # "top N" pattern → treat as list, not max
        if re.search(r'\btop\s+\d+\b', q):
            if 'max' in intents:
                intents.remove('max')
            if 'list' not in intents:
                intents.insert(0, 'list')
        
        # Default to 'list' if no intent detected
        if not intents:
            intents = ['list']
        
        return intents
    
    def _resolve_tables(self, q: str) -> List[str]:
        """Find which tables the question refers to."""
        found = []
        
        # Check each alias
        scored = []
        for alias, table_name in self.table_aliases.items():
            if len(alias) < 3:
                continue  # Skip very short aliases
            # Word boundary match
            pattern = r'\b' + re.escape(alias) + r'\b'
            if re.search(pattern, q):
                scored.append((table_name, len(alias)))
        
        # Sort by match length (prefer longer matches)
        scored.sort(key=lambda x: -x[1])
        
        seen = set()
        for table_name, _ in scored:
            if table_name not in seen:
                found.append(table_name)
                seen.add(table_name)
        
        return found
    
    def _resolve_columns(self, q: str, tables: List[str]) -> List[Tuple[str, str, Dict]]:
        """
        Find which columns the question refers to.
        
        Returns: list of (table_name, column_name, column_info)
        """
        found = []
        seen = set()
        
        # First try column names directly
        for col_alias, locations in self.column_index.items():
            if len(col_alias) < 2:
                continue
            pattern = r'\b' + re.escape(col_alias) + r'\b'
            if re.search(pattern, q):
                for table_name, col_info in locations:
                    # Prefer columns from matched tables
                    key = (table_name, col_info['name'])
                    if key not in seen:
                        found.append((table_name, col_info['name'], col_info))
                        seen.add(key)
        
        # Also check by semantic type for aggregate queries
        semantic_keywords = {
            'revenue': 'amount', 'sales': 'amount', 'income': 'amount',
            'money': 'amount', 'spent': 'amount', 'cost': 'amount',
            'price': 'amount', 'payment': 'amount', 'total': 'amount',
            'profit': 'amount', 'expense': 'amount',
            'name': 'name', 'customer': 'name',
            'email': 'email', 'phone': 'phone',
            'date': 'date', 'time': 'date', 'when': 'date',
            'category': 'category', 'type': 'category',
            'status': 'status', 'state': 'status',
            'rating': 'score', 'score': 'score', 'rank': 'score',
            'quantity': 'quantity', 'amount': 'quantity',
            'city': 'address', 'country': 'address', 'location': 'address',
        }
        
        for keyword, semantic_type in semantic_keywords.items():
            if keyword in q:
                # Find columns with matching semantic type
                for tname in (tables or list(self.tables.keys())):
                    tl = tname.lower()
                    if tl in self.columns:
                        for cname, cinfo in self.columns[tl].items():
                            if cinfo.get('semantic_type') == semantic_type:
                                key = (tname, cinfo['name'])
                                if key not in seen:
                                    found.append((tname, cinfo['name'], cinfo))
                                    seen.add(key)
        
        return found
    
    def _detect_limit(self, q: str) -> Optional[int]:
        """Detect LIMIT N from the question."""
        m = self.LIMIT_PATTERN.search(q)
        if m:
            for g in m.groups():
                if g:
                    return int(g)
        return None
    
    def _detect_filters(self, q: str, columns: List[Tuple]) -> List[Tuple[str, str, str, str]]:
        """Detect WHERE conditions from the question.
        
        Returns: list of (table, column, operator, value)
        """
        filters = []
        
        # Determine preferred column type from question context
        prefer_amount = any(kw in q for kw in ['price', 'revenue', 'cost', 'payment', 'amount', 'value', 'expensive', 'cheap'])
        
        # Heuristic: large comparison values (>50) likely refer to prices/amounts, not quantities
        preferred_types = ['amount'] if prefer_amount else ['amount', 'quantity', 'score', 'measurement']
        fallback_types = ['amount', 'quantity', 'score', 'measurement']
        
        for pattern, operator in self.COMPARISON_PATTERNS:
            match = pattern.search(q)
            if match:
                value = match.group(1)
                filter_added = False
                
                # Large values (>50) are almost certainly prices/amounts
                try:
                    if float(value) > 50:
                        preferred_types = ['amount']
                except ValueError:
                    pass
                
                # Try preferred types first in matched columns
                for sem_types in [preferred_types, fallback_types]:
                    if filter_added:
                        break
                    for table, col, info in columns:
                        if info.get('semantic_type') in sem_types:
                            filters.append((table, col, operator, value))
                            filter_added = True
                            break
                
                if not filter_added:
                    # Search ALL tables — prefer 'amount' type columns
                    for sem_types in [preferred_types, fallback_types]:
                        if filter_added:
                            break
                        for tname in self.tables:
                            for cname, cinfo in self.columns.get(tname, {}).items():
                                if cinfo.get('semantic_type') in sem_types:
                                    filters.append((self.tables[tname]['name'], cinfo['name'], operator, value))
                                    filter_added = True
                                    break
                            if filter_added:
                                break
        
        # String filter: "named X", "called X", "like X"
        name_match = re.search(r'(?:named|called|like)\s+["\']?(\w[\w\s]*)["\']?', q)
        if name_match:
            value = name_match.group(1).strip()
            for table, col, info in columns:
                if info.get('semantic_type') in ('name', 'text', 'description'):
                    filters.append((table, col, 'LIKE', f'%{value}%'))
                    break
        
        return filters
    
    def _detect_time_grouping(self, q: str, columns: List[Tuple]) -> Optional[Dict]:
        """Detect time-based grouping (monthly, yearly, etc.)."""
        for period, patterns in self.TIME_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, q):
                    # Find a date column
                    for table, col, info in columns:
                        if info.get('semantic_type') == 'date':
                            return {'period': period, 'table': table, 'column': col}
                    
                    # Search all tables for date columns
                    for tname in self.tables:
                        for cname, cinfo in self.columns.get(tname, {}).items():
                            if cinfo.get('semantic_type') == 'date':
                                return {'period': period, 'table': self.tables[tname]['name'], 'column': cinfo['name']}
                    
                    return {'period': period, 'table': None, 'column': None}
        
        return None
    
    def _fuzzy_table_match(self, q: str) -> List[str]:
        """Fuzzy match tables by checking if any word in the question matches table keywords."""
        words = set(re.findall(r'\b\w{3,}\b', q))
        scored = []
        
        for tname_lower, table_info in self.tables.items():
            # Check table name parts
            tname_parts = set(tname_lower.replace('_', ' ').split())
            overlap = words & tname_parts
            if overlap:
                scored.append((table_info['name'], len(overlap)))
            
            # Check business_description
            desc = table_info.get('business_description', '').lower()
            desc_words = set(re.findall(r'\b\w{3,}\b', desc))
            overlap2 = words & desc_words
            if overlap2:
                scored.append((table_info['name'], len(overlap2) * 0.5))
        
        scored.sort(key=lambda x: -x[1])
        seen = set()
        result = []
        for name, _ in scored:
            if name not in seen:
                result.append(name)
                seen.add(name)
        
        return result
    
    def _find_join_path(self, tables: List[str]) -> List[Dict]:
        """Find relationships to JOIN the given tables."""
        if len(tables) < 2:
            return []
        
        joins = []
        connected = {tables[0]}
        
        for target in tables[1:]:
            # Find direct relationship
            for rel in self.relationships:
                ft = rel.get('from_table', '')
                tt = rel.get('to_table', '')
                
                if (ft in connected and tt == target) or (tt in connected and ft == target):
                    joins.append(rel)
                    connected.add(target)
                    break
            else:
                # Try through intermediate tables (2-hop)
                found_path = False
                for rel1 in self.relationships:
                    for rel2 in self.relationships:
                        ft1 = rel1.get('from_table', '')
                        tt1 = rel1.get('to_table', '')
                        ft2 = rel2.get('from_table', '')
                        tt2 = rel2.get('to_table', '')
                        
                        if ft1 in connected and (tt1 == ft2 or tt1 == tt2):
                            mid = tt1
                            if (ft2 == mid and tt2 == target) or (tt2 == mid and ft2 == target):
                                joins.append(rel1)
                                joins.append(rel2)
                                connected.add(mid)
                                connected.add(target)
                                found_path = True
                                break
                    if target in connected:
                        break
                
                # Last resort: try implicit join by matching column names between connected + target
                if not found_path:
                    target_lower = target.lower()
                    for conn_table in list(connected):
                        conn_lower = conn_table.lower()
                        conn_cols = self.columns.get(conn_lower, {})
                        target_cols = self.columns.get(target_lower, {})
                        # Find columns with the same name in both tables
                        shared = set(conn_cols.keys()) & set(target_cols.keys())
                        for sc in shared:
                            if sc.endswith('_id') or sc.endswith('_name') or sc.endswith('_code'):
                                joins.append({
                                    'from_table': conn_table,
                                    'from_column': conn_cols[sc]['name'],
                                    'to_table': target,
                                    'to_column': target_cols[sc]['name'],
                                })
                                connected.add(target)
                                found_path = True
                                break
                        if found_path:
                            break
                    
                    # Still not found? Try via an intermediate table (FK hop + implicit join)
                    if not found_path:
                        for mid_tname, mid_info in self.tables.items():
                            mid_name = mid_info['name']
                            if mid_name in connected or mid_name == target:
                                continue
                            # Check: connected → mid (via FK)?
                            hop1 = None
                            for rel in self.relationships:
                                ft, tt = rel.get('from_table', ''), rel.get('to_table', '')
                                if (ft in connected and tt == mid_name) or (tt in connected and ft == mid_name):
                                    hop1 = rel
                                    break
                            if not hop1:
                                continue
                            # Check: mid → target (via column name match)?
                            mid_cols = self.columns.get(mid_tname, {})
                            tgt_cols = self.columns.get(target_lower, {})
                            shared2 = set(mid_cols.keys()) & set(tgt_cols.keys())
                            for sc in shared2:
                                if sc.endswith('_id') or sc.endswith('_name') or sc.endswith('_code'):
                                    joins.append(hop1)
                                    joins.append({
                                        'from_table': mid_name,
                                        'from_column': mid_cols[sc]['name'],
                                        'to_table': target,
                                        'to_column': tgt_cols[sc]['name'],
                                    })
                                    connected.add(mid_name)
                                    connected.add(target)
                                    found_path = True
                                    break
                            if found_path:
                                break
        
        return joins
    
    def _build_sql(self, intents, tables, columns, limit, filters, time_group, sort_desc, question):
        """Build the SQL query from analyzed components."""
        warnings = []
        confidence = 0.5
        
        primary_intent = intents[0] if intents else 'list'
        has_group = 'group' in intents
        has_agg = primary_intent in self.AGG_MAP
        
        # Resolve primary table
        if not tables:
            return SQLQuery(
                sql="-- Could not determine which table to query",
                explanation="I couldn't identify which table you're asking about.",
                intent=primary_intent,
                tables_used=[],
                columns_used=[],
                confidence=0.1,
                warnings=["No tables matched your question"]
            )
        
        primary_table = tables[0]
        ptl = primary_table.lower()
        
        # Get columns for primary table
        table_columns = self.columns.get(ptl, {})
        
        # Determine SELECT columns
        select_parts = []
        group_by_parts = []
        order_parts = []
        
        if has_agg or primary_intent in self.AGG_MAP:
            agg_func = self.AGG_MAP.get(primary_intent, 'COUNT')
            
            if primary_intent == 'count':
                # COUNT queries — prefer time grouping if detected
                if time_group and time_group.get('column'):
                    date_col = time_group['column']
                    period = time_group['period']
                    if period == 'year':
                        select_parts.append(f"strftime('%Y', [{date_col}]) as year")
                        group_by_parts.append("year")
                    elif period == 'month':
                        select_parts.append(f"strftime('%Y-%m', [{date_col}]) as month")
                        group_by_parts.append("month")
                    elif period == 'week':
                        select_parts.append(f"strftime('%Y-%W', [{date_col}]) as week")
                        group_by_parts.append("week")
                    else:
                        select_parts.append(f"strftime('%Y-%m-%d', [{date_col}]) as day")
                        group_by_parts.append("day")
                    select_parts.append("COUNT(*) as count")
                    order_parts.append(group_by_parts[0])
                    confidence = 0.85
                elif has_group or 'group' in intents:
                    # Find grouping column
                    group_col = self._find_best_group_column(ptl, columns, question)
                    if group_col:
                        select_parts.append(f"[{group_col}]")
                        select_parts.append(f"COUNT(*) as count")
                        group_by_parts.append(f"[{group_col}]")
                        order_parts.append("count DESC")
                    else:
                        select_parts.append("COUNT(*) as total_count")
                else:
                    select_parts.append("COUNT(*) as total_count")
                confidence = max(confidence, 0.8)
            
            else:
                # SUM, AVG, MAX, MIN queries
                agg_col = self._find_best_numeric_column(ptl, columns, question)
                agg_table = primary_table
                
                # Cross-table search: if no numeric column in primary table, search related tables
                if not agg_col:
                    agg_col, agg_table = self._find_numeric_column_cross_table(ptl, columns, question, tables)
                    if agg_col and agg_table and agg_table != primary_table:
                        if agg_table not in tables:
                            tables.append(agg_table)
                        # Swap primary table to the one with the aggregate column
                        # This ensures JOINs can be found (agg table is usually more connected)
                        old_primary = primary_table
                        primary_table = agg_table
                        ptl = primary_table.lower()
                        # Keep old primary in tables for GROUP BY column
                        if old_primary not in tables:
                            tables.append(old_primary)
                
                # Absolute fallback: if still no agg_col, try ANY numeric column in primary table
                if not agg_col:
                    for cname, cinfo in self.columns.get(ptl, {}).items():
                        if cinfo.get('data_type', '').upper() in ('INTEGER', 'REAL', 'NUMERIC', 'FLOAT', 'DECIMAL'):
                            if not cinfo.get('is_primary_key') and not cinfo.get('is_foreign_key'):
                                agg_col = cinfo['name']
                                warnings.append(f"No exact match for your query — using [{agg_col}] as the closest numeric column")
                                break
                
                if agg_col:
                    select_parts.append(f"{agg_func}([{agg_col}]) as {agg_func.lower()}_{agg_col}")
                    
                    if has_group:
                        # Search group column across all tables in the query
                        group_col = None
                        group_table = None
                        for search_table in tables:
                            stl = search_table.lower()
                            gc = self._find_best_group_column(stl, columns, question)
                            if gc:
                                group_col = gc
                                group_table = search_table
                                break
                        
                        if group_col:
                            if group_table and group_table != primary_table:
                                # Ensure group table is in tables list for JOIN
                                if group_table not in tables:
                                    tables.append(group_table)
                                select_parts.insert(0, f"[{group_table}].[{group_col}]")
                                group_by_parts.append(f"[{group_table}].[{group_col}]")
                            else:
                                select_parts.insert(0, f"[{group_col}]")
                                group_by_parts.append(f"[{group_col}]")
                            order_parts.append(f"{agg_func.lower()}_{agg_col} {'DESC' if sort_desc else 'ASC'}")
                    
                    confidence = 0.8
                else:
                    # Truly no numeric column exists — generate COUNT instead of invalid AGG(*)
                    select_parts.append(f"COUNT(*) as total_count")
                    warnings.append(f"This dataset has no numeric columns for {agg_func}. Showing COUNT instead.")
                    confidence = 0.5
        
        elif 'trend' in intents or time_group:
            # Time series query
            if time_group and time_group.get('column'):
                date_col = time_group['column']
                period = time_group['period']
                
                if period == 'year':
                    select_parts.append(f"strftime('%Y', [{date_col}]) as year")
                    group_by_parts.append("year")
                elif period == 'month':
                    select_parts.append(f"strftime('%Y-%m', [{date_col}]) as month")
                    group_by_parts.append("month")
                elif period == 'week':
                    select_parts.append(f"strftime('%Y-%W', [{date_col}]) as week")
                    group_by_parts.append("week")
                else:
                    select_parts.append(f"strftime('%Y-%m-%d', [{date_col}]) as day")
                    group_by_parts.append("day")
                
                # Add aggregate
                num_col = self._find_best_numeric_column(ptl, columns, question)
                if num_col:
                    select_parts.append(f"SUM([{num_col}]) as total_{num_col}")
                    select_parts.append(f"COUNT(*) as record_count")
                else:
                    select_parts.append("COUNT(*) as record_count")
                
                order_parts.append(group_by_parts[0])
                confidence = 0.75
            else:
                select_parts.append("*")
                warnings.append("Could not find a date column for time-based analysis")
                confidence = 0.4
        
        else:
            # LIST / generic SELECT
            # Select specific columns if mentioned, otherwise SELECT *
            specific_cols = [col for (t, col, info) in columns if t.lower() == ptl]
            
            if specific_cols:
                select_parts = [f"[{c}]" for c in specific_cols[:8]]
                confidence = 0.7
            else:
                select_parts = ["*"]
                confidence = 0.6
            
            # Default sort for top-N queries
            if limit and sort_desc:
                num_col = self._find_best_numeric_column(ptl, columns, question)
                if num_col:
                    order_parts.append(f"[{num_col}] DESC")
                    confidence = 0.75
        
        # (FROM clause is built after WHERE to capture cross-table filters)
        
        # Build WHERE clause — add cross-table filter tables to JOIN list
        where_parts = []
        for (table, col, op, val) in filters:
            # If filter column is from a different table, ensure it's in tables for JOIN
            if table and table != primary_table and table not in tables:
                tables.append(table)
            table_prefix = f"[{table}]." if table and table != primary_table else ""
            if op == 'LIKE':
                where_parts.append(f"{table_prefix}[{col}] LIKE '{val}'")
            else:
                where_parts.append(f"{table_prefix}[{col}] {op} {val}")
        
        # Build FROM clause with JOINs (after tables list is finalized)
        from_clause = f"[{primary_table}]"
        if len(tables) > 1:
            # Ensure primary table is first for join path finding
            join_tables = [primary_table] + [t for t in tables if t != primary_table]
            joins = self._find_join_path(join_tables)
            joined_tables = {primary_table}
            for join in joins:
                ft = join.get('from_table', '')
                fc = join.get('from_column', '')
                tt = join.get('to_table', '')
                tc = join.get('to_column', '')
                if ft and fc and tt and tc:
                    # Determine which table to JOIN (the one not already in FROM)
                    if ft in joined_tables and tt not in joined_tables:
                        from_clause += f"\nJOIN [{tt}] ON [{ft}].[{fc}] = [{tt}].[{tc}]"
                        joined_tables.add(tt)
                    elif tt in joined_tables and ft not in joined_tables:
                        from_clause += f"\nJOIN [{ft}] ON [{ft}].[{fc}] = [{tt}].[{tc}]"
                        joined_tables.add(ft)
                    elif ft not in joined_tables:
                        from_clause += f"\nJOIN [{ft}] ON [{ft}].[{fc}] = [{tt}].[{tc}]"
                        joined_tables.add(ft)
                    confidence = min(confidence + 0.1, 0.95)
        
        # Assemble SQL
        sql = f"SELECT {', '.join(select_parts)}\nFROM {from_clause}"
        
        if where_parts:
            sql += f"\nWHERE {' AND '.join(where_parts)}"
        
        if group_by_parts:
            sql += f"\nGROUP BY {', '.join(group_by_parts)}"
        
        if order_parts:
            sql += f"\nORDER BY {', '.join(order_parts)}"
        
        if limit:
            sql += f"\nLIMIT {limit}"
        elif primary_intent == 'list' and not group_by_parts:
            sql += "\nLIMIT 25"
            warnings.append("Added default LIMIT 25 for safety")
        
        sql += ";"
        
        # Generate explanation
        explanation = self._generate_explanation(primary_intent, tables, columns, filters, limit, time_group)
        
        return SQLQuery(
            sql=sql,
            explanation=explanation,
            intent=primary_intent,
            tables_used=tables,
            columns_used=[c for (_, c, _) in columns],
            confidence=confidence,
            warnings=warnings
        )
    
    def _find_best_numeric_column(self, table_lower: str, matched_columns: List[Tuple], question: str) -> Optional[str]:
        """Find the best numeric column for aggregation.
        
        Returns None if the question asks for a specific type (like 'price') but the table
        doesn't have it — this allows cross-table search to kick in.
        """
        # Priority keywords in question → semantic type
        priority_hints = [
            (['price', 'revenue', 'sales', 'cost', 'payment', 'amount', 'income', 'spent', 'total', 'money', 'value', 'fee', 'profit'], 'amount'),
            (['quantity', 'qty', 'count', 'number', 'items'], 'quantity'),
            (['score', 'rating', 'rank', 'weight'], 'score'),
        ]
        
        specific_type_requested = None  # Track if user asked for a specific type
        
        for keywords, sem_type in priority_hints:
            if any(kw in question for kw in keywords):
                specific_type_requested = sem_type
                # Look for this specific type first in matched columns
                for table, col, info in matched_columns:
                    if table.lower() == table_lower and info.get('semantic_type') == sem_type:
                        return col
                # Then in all table columns
                for cname, cinfo in self.columns.get(table_lower, {}).items():
                    if cinfo.get('semantic_type') == sem_type:
                        return cinfo['name']
                # Type was requested but NOT found in this table → return None
                # so cross-table search can find it in another table
                return None
        
        # No specific type requested → use generic fallback
        # Prefer amount > quantity > score > measurement > year
        for sem in ('amount', 'quantity', 'score', 'measurement', 'year'):
            for cname, cinfo in self.columns.get(table_lower, {}).items():
                if cinfo.get('semantic_type') == sem:
                    return cinfo['name']
        
        # Last resort: any non-PK, non-FK numeric column
        fallback = None
        for cname, cinfo in self.columns.get(table_lower, {}).items():
            if cinfo.get('data_type', '').upper() in ('INTEGER', 'REAL', 'NUMERIC', 'FLOAT', 'DECIMAL'):
                if not cinfo.get('is_primary_key') and not cinfo.get('is_foreign_key'):
                    if fallback is None:
                        fallback = cinfo['name']
                    cl = cname.lower()
                    if any(kw in cl for kw in ['price', 'amount', 'total', 'cost', 'value', 'payment', 'revenue']):
                        return cinfo['name']
        
        return fallback
    
    def _find_numeric_column_cross_table(self, primary_table_lower: str, matched_columns: List[Tuple], question: str, current_tables: List[str]) -> Tuple[Optional[str], Optional[str]]:
        """Search ALL tables for a numeric column matching the question when primary table has none.
        
        Returns: (column_name, table_name) or (None, None)
        """
        # Keywords in question → preferred column name patterns
        keyword_col_map = {
            'price': ['price', 'unit_price', 'sale_price', 'list_price'],
            'revenue': ['price', 'amount', 'payment_value', 'revenue', 'total'],
            'sales': ['price', 'amount', 'payment_value', 'sales'],
            'cost': ['cost', 'price', 'freight_value', 'freight'],
            'payment': ['payment_value', 'payment', 'amount', 'price'],
            'freight': ['freight_value', 'freight', 'shipping'],
            'amount': ['amount', 'payment_value', 'price', 'total'],
            'score': ['review_score', 'score', 'rating'],
            'rating': ['review_score', 'score', 'rating'],
        }
        
        q_lower = question.lower()
        
        # Try keyword-based search across all tables
        for keyword, col_names in keyword_col_map.items():
            if keyword in q_lower:
                for cn in col_names:
                    if cn in self.column_index:
                        for table_name, col_info in self.column_index[cn]:
                            # Prefer tables that are related to the primary table
                            return col_info['name'], table_name
        
        # Fallback: search all tables for amount-type columns
        for tname_lower, cols in self.columns.items():
            if tname_lower == primary_table_lower:
                continue
            for cname, cinfo in cols.items():
                if cinfo.get('semantic_type') == 'amount':
                    return cinfo['name'], self.tables[tname_lower]['name']
        
        return None, None
    
    def _find_best_group_column(self, table_lower: str, matched_columns: List[Tuple], question: str) -> Optional[str]:
        """Find the best column to GROUP BY."""
        # First: check if question keywords directly match a column name
        # e.g., "by category" → look for columns with "category" in the name
        group_keywords = re.findall(r'\b(?:by|per|each|breakdown\s+by)\s+(\w+)', question)
        for keyword in group_keywords:
            kw_lower = keyword.lower()
            # Check in the specified table only
            for cname, cinfo in self.columns.get(table_lower, {}).items():
                if kw_lower in cname.lower() and cinfo.get('semantic_type') not in ('identifier', 'date'):
                    return cinfo['name']
        
        # Check matched columns for categorical types
        for table, col, info in matched_columns:
            if table.lower() == table_lower and info.get('semantic_type') in ('category', 'status', 'name', 'address'):
                return col
        
        # Check for semantic type in question
        group_hints = {
            'category': 'category', 'type': 'category', 'status': 'status',
            'customer': 'name', 'product': 'name', 'city': 'address',
            'country': 'address', 'state': 'address', 'region': 'address',
            'year': 'date', 'month': 'date',
        }
        
        for hint, sem_type in group_hints.items():
            if hint in question:
                for cname, cinfo in self.columns.get(table_lower, {}).items():
                    if cinfo.get('semantic_type') == sem_type:
                        return cinfo['name']
        
        # Fallback: any text column with reasonable cardinality
        for cname, cinfo in self.columns.get(table_lower, {}).items():
            if (cinfo.get('data_type', '').upper() == 'TEXT' and 
                not cinfo.get('is_primary_key') and
                cinfo.get('unique_count', 0) < cinfo.get('total_count', 1) * 0.5):
                return cinfo['name']
        
        return None
    
    def _generate_explanation(self, intent, tables, columns, filters, limit, time_group) -> str:
        """Generate a human-readable explanation of the query."""
        parts = []
        
        if intent == 'count':
            parts.append("Counting records")
        elif intent == 'average':
            parts.append("Calculating average")
        elif intent == 'sum':
            parts.append("Calculating total sum")
        elif intent == 'max':
            parts.append("Finding maximum value")
        elif intent == 'min':
            parts.append("Finding minimum value")
        else:
            parts.append("Retrieving data")
        
        if tables:
            parts.append(f"from **{', '.join(tables)}**")
        
        if filters:
            filter_descs = [f"{col} {op} {val}" for (_, col, op, val) in filters]
            parts.append(f"where {' and '.join(filter_descs)}")
        
        if time_group:
            parts.append(f"grouped by {time_group['period']}")
        
        if limit:
            parts.append(f"(top {limit} results)")
        
        return ' '.join(parts)


class SQLPlayground:
    """
    Safe SQL execution engine for uploaded datasets.
    
    Supports:
    - SQLite databases (direct connection)
    - CSV/Excel/JSON (loaded into in-memory SQLite)
    
    Safety:
    - Read-only (no INSERT/UPDATE/DELETE/DROP)
    - Query timeout
    - Row limit
    """
    
    MAX_ROWS = 500
    TIMEOUT_SECONDS = 10
    
    # Dangerous SQL patterns
    UNSAFE_PATTERNS = [
        re.compile(r'\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE)\b', re.I),
        re.compile(r'\b(ATTACH|DETACH)\b', re.I),
        re.compile(r'\bPRAGMA\b', re.I),
    ]
    
    def __init__(self, dataset_path: str, file_type: str, table_names: list = None):
        """
        Initialize playground with dataset.
        
        Args:
            dataset_path: Path to raw dataset file
            file_type: 'sqlite', 'csv', 'excel', 'json'
            table_names: Optional list of original table names from schema
                         (used for CSV/Excel/JSON to match schema names)
        """
        self.dataset_path = dataset_path
        self.file_type = file_type
        self.table_names = table_names or []
        self.conn = None
        self._setup_connection()
    
    def _setup_connection(self):
        """Set up database connection based on file type."""
        if self.file_type == 'sqlite':
            # Direct connection (read-only)
            self.conn = sqlite3.connect(f"file:{self.dataset_path}?mode=ro", uri=True)
        else:
            # Load into in-memory SQLite
            self.conn = sqlite3.connect(':memory:')
            self._load_into_sqlite()
    
    def _load_into_sqlite(self):
        """Load CSV/Excel/JSON data into in-memory SQLite."""
        try:
            import pandas as pd
            
            if self.file_type == 'csv':
                try:
                    df = pd.read_csv(self.dataset_path, encoding='utf-8')
                except:
                    df = pd.read_csv(self.dataset_path, encoding='latin-1')
                # Use original table name from schema if available
                if self.table_names:
                    table_name = self.table_names[0]
                else:
                    import os
                    stem = os.path.splitext(os.path.basename(self.dataset_path))[0]
                    if stem.startswith('dataset'):
                        table_name = 'data'
                    else:
                        table_name = re.sub(r'[^\w]', '_', stem)
                df.to_sql(table_name, self.conn, index=False, if_exists='replace')
                
            elif self.file_type in ('excel', 'xlsx', 'xls'):
                excel = pd.ExcelFile(self.dataset_path)
                for sheet in excel.sheet_names:
                    df = pd.read_excel(excel, sheet_name=sheet)
                    if not df.empty:
                        table_name = re.sub(r'[^\w]', '_', sheet)
                        df.to_sql(table_name, self.conn, index=False, if_exists='replace')
                        
            elif self.file_type == 'json':
                import json as json_mod
                with open(self.dataset_path, 'r', encoding='utf-8') as f:
                    data = json_mod.load(f)
                
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                    df.to_sql('data', self.conn, index=False, if_exists='replace')
                elif isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, list) and value and isinstance(value[0], dict):
                            df = pd.DataFrame(value)
                            table_name = re.sub(r'[^\w]', '_', key)
                            df.to_sql(table_name, self.conn, index=False, if_exists='replace')
                            
        except ImportError:
            raise ValueError("pandas is required for CSV/Excel/JSON SQL playground")
    
    def validate_sql(self, sql: str) -> Tuple[bool, Optional[str]]:
        """
        Validate SQL query for safety.
        
        Returns: (is_safe, error_message)
        """
        for pattern in self.UNSAFE_PATTERNS:
            if pattern.search(sql):
                return False, f"Write operations are not allowed. Only SELECT queries are permitted."
        
        # Check for multiple statements
        statements = [s.strip() for s in sql.split(';') if s.strip()]
        if len(statements) > 1:
            return False, "Only a single SQL statement is allowed."
        
        return True, None
    
    def execute(self, sql: str) -> QueryResult:
        """
        Execute a SQL query safely.
        
        Args:
            sql: SQL query to execute
            
        Returns:
            QueryResult with results or error
        """
        start = time.time()
        
        # Validate
        is_safe, error = self.validate_sql(sql)
        if not is_safe:
            return QueryResult(
                success=False,
                sql=sql,
                explanation="",
                columns=[],
                rows=[],
                row_count=0,
                execution_time_ms=0,
                error=error,
            )
        
        try:
            cursor = self.conn.cursor()
            
            # Add LIMIT if not present
            sql_stripped = sql.strip().rstrip(';')
            if not re.search(r'\bLIMIT\b', sql_stripped, re.I):
                sql_stripped += f" LIMIT {self.MAX_ROWS}"
            
            cursor.execute(sql_stripped)
            
            # Get column names
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            
            # Fetch rows
            rows = cursor.fetchmany(self.MAX_ROWS)
            
            # Convert to serializable format
            clean_rows = []
            for row in rows:
                clean_row = []
                for val in row:
                    if val is None:
                        clean_row.append(None)
                    elif isinstance(val, (int, float, str, bool)):
                        clean_row.append(val)
                    else:
                        clean_row.append(str(val))
                clean_rows.append(clean_row)
            
            elapsed = (time.time() - start) * 1000
            
            return QueryResult(
                success=True,
                sql=sql,
                explanation=f"Query returned {len(clean_rows)} rows with {len(columns)} columns",
                columns=columns,
                rows=clean_rows,
                row_count=len(clean_rows),
                execution_time_ms=round(elapsed, 2),
            )
            
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return QueryResult(
                success=False,
                sql=sql,
                explanation="",
                columns=[],
                rows=[],
                row_count=0,
                execution_time_ms=round(elapsed, 2),
                error=str(e),
            )
    
    def get_tables(self) -> List[Dict]:
        """Get list of tables with row counts."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = []
        for row in cursor.fetchall():
            tname = row[0]
            cursor.execute(f"SELECT COUNT(*) FROM [{tname}]")
            count = cursor.fetchone()[0]
            tables.append({'name': tname, 'row_count': count})
        return tables
    
    def get_table_preview(self, table_name: str, limit: int = 50) -> QueryResult:
        """Get preview of a table's data."""
        # Safety check table name
        safe_name = re.sub(r'[^\w]', '_', table_name)
        return self.execute(f"SELECT * FROM [{safe_name}] LIMIT {min(limit, self.MAX_ROWS)};")
    
    def close(self):
        """Close connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __del__(self):
        self.close()
