"""
DataMind AI - Embedding Search & RAG Engine
============================================
Enterprise-grade Retrieval-Augmented Generation system for database documentation.

Architecture:
┌─────────────────────────────────────────────────────────┐
│                    RAG Pipeline                         │
│                                                         │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │ Chunker  │──▶│ Vectorizer   │──▶│ Vector Store   │  │
│  │ (Schema  │   │ (TF-IDF /    │   │ (Cosine Index) │  │
│  │  Splitter│   │  BM25 /      │   │                │  │
│  │  )       │   │  Embeddings) │   │                │  │
│  └──────────┘   └──────────────┘   └────────┬───────┘  │
│                                              │          │
│  ┌──────────┐   ┌──────────────┐   ┌────────▼───────┐  │
│  │ Response │◀──│ Re-Ranker    │◀──│ Similarity     │  │
│  │ Generator│   │ (MMR + BM25) │   │ Search         │  │
│  └──────────┘   └──────────────┘   └────────────────┘  │
└─────────────────────────────────────────────────────────┘

Features:
- Multi-strategy chunking (schema, column, relationship, quality)
- TF-IDF vectorization with n-gram support
- BM25 scoring for keyword relevance
- Cosine similarity search
- Maximal Marginal Relevance (MMR) for diverse results
- Context-aware response generation
- Persistent vector index per dataset
"""

import os
import json
import math
import re
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
from datetime import datetime

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Chunk:
    """A chunk of information from the database analysis."""
    chunk_id: str
    content: str               # Human readable text
    chunk_type: str            # table_overview, column_detail, relationship, quality, stat, query
    metadata: Dict = field(default_factory=dict)
    source_table: str = ""
    source_column: str = ""
    keywords: List[str] = field(default_factory=list)
    importance_score: float = 1.0

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict) -> 'Chunk':
        return Chunk(**d)


@dataclass
class SearchResult:
    """A single search result."""
    chunk: Chunk
    score: float
    rank: int
    match_type: str = "semantic"   # semantic, keyword, hybrid


@dataclass
class QueryResponse:
    """Complete response to a user query."""
    query: str
    answer: str
    relevant_chunks: List[SearchResult]
    confidence: float
    processing_time_ms: float
    total_chunks_searched: int


# ============================================================================
# CHUNKING ENGINE - Splits analysis results into searchable chunks
# ============================================================================

class SchemaChunker:
    """
    Intelligent chunking of database analysis results.
    
    Chunking strategies:
    1. Table-level overview chunks
    2. Column-level detail chunks  
    3. Relationship chunks
    4. Quality issue chunks
    5. Statistical summary chunks
    6. Sample query chunks
    7. Cross-table insight chunks
    """

    def __init__(self):
        self._chunk_counter = 0

    def _make_id(self, prefix: str) -> str:
        self._chunk_counter += 1
        return f"{prefix}_{self._chunk_counter:04d}"

    def chunk_analysis(self, results: Dict) -> List[Chunk]:
        """
        Convert full analysis results into searchable chunks.
        
        Each chunk is a self-contained piece of information
        that can be retrieved independently.
        """
        chunks = []
        db_name = results.get('database_name', 'Database')
        
        # 1. Database overview chunk
        chunks.append(self._create_db_overview_chunk(results, db_name))
        
        # 2. Table-level chunks
        for table in results.get('tables', []):
            chunks.extend(self._create_table_chunks(table, db_name))
        
        # 3. Relationship chunks
        for rel in results.get('relationships', []):
            chunks.append(self._create_relationship_chunk(rel, db_name))
        
        # 4. Quality issue chunks
        for issue in results.get('quality_issues', []):
            chunks.append(self._create_quality_chunk(issue, db_name))
        
        # 5. Sample query chunks
        for query in results.get('sample_queries', []):
            chunks.append(self._create_query_chunk(query, db_name))
        
        # 6. Cross-table insight chunks
        chunks.extend(self._create_cross_table_chunks(results, db_name))
        
        # 7. Summary statistics chunks
        chunks.extend(self._create_stat_chunks(results, db_name))
        
        return chunks

    def _create_db_overview_chunk(self, results: Dict, db_name: str) -> Chunk:
        summary = results.get('summary', {})
        tables = results.get('tables', [])
        table_names = [t.get('name', '') for t in tables]
        
        content = (
            f"The database '{db_name}' is a {results.get('file_type', 'relational')} database "
            f"containing {summary.get('total_tables', 0)} tables with "
            f"{summary.get('total_columns', 0)} total columns and "
            f"{summary.get('total_rows', 0)} total records. "
            f"Tables: {', '.join(table_names)}. "
            f"It has {summary.get('total_relationships', 0)} relationships detected "
            f"and a data quality score of {summary.get('quality_score', 0)}/100. "
            f"{summary.get('quality_issues', 0)} quality issues were found."
        )
        
        return Chunk(
            chunk_id=self._make_id("db"),
            content=content,
            chunk_type="db_overview",
            metadata=summary,
            keywords=["database", "overview", "summary", "tables", "schema", db_name.lower()] + 
                     [t.lower() for t in table_names],
            importance_score=2.0
        )

    def _create_table_chunks(self, table: Dict, db_name: str) -> List[Chunk]:
        chunks = []
        table_name = table.get('name', 'Unknown')
        columns = table.get('columns', [])
        row_count = table.get('row_count', 0)
        table_type = table.get('table_type', 'unknown')
        desc = table.get('business_description', '')
        pks = table.get('primary_keys', [])
        
        # Table overview chunk
        col_names = [c.get('name', '') for c in columns]
        pk_text = f"Primary keys: {', '.join(pks)}. " if pks else ""
        fk_cols = [c.get('name', '') for c in columns if c.get('is_foreign_key')]
        fk_text = f"Foreign keys: {', '.join(fk_cols)}. " if fk_cols else ""
        
        overview_content = (
            f"Table '{table_name}' is a {table_type} table with {len(columns)} columns "
            f"and {row_count:,} rows. {pk_text}{fk_text}"
            f"Description: {desc}. "
            f"Columns: {', '.join(col_names)}."
        )
        
        chunks.append(Chunk(
            chunk_id=self._make_id("tbl"),
            content=overview_content,
            chunk_type="table_overview",
            source_table=table_name,
            metadata={"row_count": row_count, "column_count": len(columns), "table_type": table_type},
            keywords=["table", table_name.lower(), table_type, "schema", "structure"] + 
                     [c.lower() for c in col_names],
            importance_score=1.5
        ))
        
        # Individual column chunks
        for col in columns:
            col_name = col.get('name', '')
            dtype = col.get('data_type', 'unknown')
            nullable = col.get('nullable', False)
            is_pk = col.get('is_primary_key', False)
            is_fk = col.get('is_foreign_key', False)
            refs = col.get('references', '')
            semantic = col.get('semantic_type', '')
            col_desc = col.get('business_description', '')
            null_pct = col.get('null_percentage', 0)
            unique_count = col.get('unique_count', 0)
            total_count = col.get('total_count', 0)
            samples = col.get('sample_values', [])
            
            role_parts = []
            if is_pk: role_parts.append("primary key")
            if is_fk: role_parts.append(f"foreign key referencing {refs}" if refs else "foreign key")
            if semantic: role_parts.append(f"semantic type: {semantic}")
            role_text = ". ".join(role_parts) + "." if role_parts else ""
            
            sample_text = ""
            if samples:
                sample_vals = [str(s) for s in samples[:5]]
                sample_text = f" Sample values: {', '.join(sample_vals)}."
            
            null_text = f" {null_pct}% null values." if null_pct > 0 else " No null values."
            uniqueness = ""
            if total_count > 0:
                uniq_pct = round(unique_count / total_count * 100, 1)
                uniqueness = f" {unique_count:,} unique values ({uniq_pct}% uniqueness)."
            
            col_content = (
                f"Column '{col_name}' in table '{table_name}': "
                f"data type {dtype}, {'nullable' if nullable else 'not nullable'}. "
                f"{role_text} {col_desc}{null_text}{uniqueness}{sample_text}"
            )
            
            chunks.append(Chunk(
                chunk_id=self._make_id("col"),
                content=col_content,
                chunk_type="column_detail",
                source_table=table_name,
                source_column=col_name,
                metadata={
                    "data_type": dtype, "is_pk": is_pk, "is_fk": is_fk,
                    "null_percentage": null_pct, "unique_count": unique_count
                },
                keywords=[col_name.lower(), table_name.lower(), dtype.lower(), 
                         "column", "field"] + 
                         (["primary key", "pk"] if is_pk else []) +
                         (["foreign key", "fk", "reference"] if is_fk else []) +
                         ([semantic.lower()] if semantic else []),
                importance_score=1.0
            ))
        
        return chunks

    def _create_relationship_chunk(self, rel: Dict, db_name: str) -> Chunk:
        from_table = rel.get('from_table', '')
        to_table = rel.get('to_table', '')
        from_col = rel.get('from_column', '')
        to_col = rel.get('to_column', '')
        rel_type = rel.get('relationship_type', 'related')
        confidence = rel.get('confidence', 0)
        
        content = (
            f"Relationship: '{from_table}.{from_col}' → '{to_table}.{to_col}'. "
            f"Type: {rel_type}. Confidence: {confidence}%. "
            f"Table '{from_table}' references table '{to_table}' "
            f"through column '{from_col}' linking to '{to_col}'."
        )
        
        return Chunk(
            chunk_id=self._make_id("rel"),
            content=content,
            chunk_type="relationship",
            metadata=rel,
            keywords=["relationship", "join", "foreign key", "link", "reference",
                      from_table.lower(), to_table.lower(), from_col.lower(), to_col.lower()],
            importance_score=1.3
        )

    def _create_quality_chunk(self, issue: Dict, db_name: str) -> Chunk:
        table = issue.get('table', '')
        column = issue.get('column', '')
        issue_type = issue.get('issue_type', '')
        severity = issue.get('severity', '')
        description = issue.get('description', '')
        recommendation = issue.get('recommendation', '')
        
        content = (
            f"Data quality issue in '{table}.{column}': {issue_type} ({severity} severity). "
            f"{description} "
            f"Recommendation: {recommendation}"
        )
        
        return Chunk(
            chunk_id=self._make_id("qual"),
            content=content,
            chunk_type="quality",
            source_table=table,
            source_column=column,
            metadata=issue,
            keywords=["quality", "issue", "problem", severity, issue_type.lower(),
                      table.lower(), column.lower()],
            importance_score=1.2
        )

    def _create_query_chunk(self, query: Dict, db_name: str) -> Chunk:
        title = query.get('title', '')
        desc = query.get('description', '')
        sql = query.get('sql', '')
        
        content = (
            f"Sample query - {title}: {desc}. "
            f"SQL: {sql}"
        )
        
        return Chunk(
            chunk_id=self._make_id("qry"),
            content=content,
            chunk_type="query",
            metadata=query,
            keywords=["query", "sql", "select", "join", "example"] + 
                     re.findall(r'\b[a-z_]+\b', sql.lower()),
            importance_score=0.8
        )

    def _create_cross_table_chunks(self, results: Dict, db_name: str) -> List[Chunk]:
        """Create chunks that describe cross-table patterns."""
        chunks = []
        tables = results.get('tables', [])
        relationships = results.get('relationships', [])
        
        if not tables:
            return chunks
        
        # Data type distribution across all tables
        type_counts = defaultdict(int)
        for table in tables:
            for col in table.get('columns', []):
                type_counts[col.get('data_type', 'unknown')] += 1
        
        if type_counts:
            type_desc = ", ".join([f"{count} {dtype}" for dtype, count in 
                                   sorted(type_counts.items(), key=lambda x: -x[1])])
            chunks.append(Chunk(
                chunk_id=self._make_id("xref"),
                content=f"Data type distribution across all tables: {type_desc}.",
                chunk_type="cross_table",
                keywords=["data type", "distribution", "schema", "column types"],
                importance_score=0.7
            ))
        
        # Largest tables
        sorted_tables = sorted(tables, key=lambda t: t.get('row_count', 0), reverse=True)
        if sorted_tables:
            largest = sorted_tables[0]
            smallest = sorted_tables[-1]
            chunks.append(Chunk(
                chunk_id=self._make_id("xref"),
                content=(
                    f"Largest table: '{largest.get('name', '')}' with "
                    f"{largest.get('row_count', 0):,} rows. "
                    f"Smallest table: '{smallest.get('name', '')}' with "
                    f"{smallest.get('row_count', 0):,} rows."
                ),
                chunk_type="cross_table",
                keywords=["largest", "smallest", "rows", "size", "table"],
                importance_score=0.8
            ))
        
        # Tables with most relationships
        rel_count = defaultdict(int)
        for rel in relationships:
            rel_count[rel.get('from_table', '')] += 1
            rel_count[rel.get('to_table', '')] += 1
        
        if rel_count:
            most_connected = max(rel_count, key=rel_count.get)
            chunks.append(Chunk(
                chunk_id=self._make_id("xref"),
                content=(
                    f"Most connected table: '{most_connected}' with "
                    f"{rel_count[most_connected]} relationships. "
                    f"This table is a central hub in the database schema."
                ),
                chunk_type="cross_table",
                keywords=["connected", "relationships", "hub", "central", most_connected.lower()],
                importance_score=0.9
            ))
        
        # Null analysis across tables
        high_null_cols = []
        for table in tables:
            for col in table.get('columns', []):
                null_pct = col.get('null_percentage', 0)
                if null_pct > 20:
                    high_null_cols.append((table.get('name', ''), col.get('name', ''), null_pct))
        
        if high_null_cols:
            null_text = "; ".join([f"{t}.{c} ({p}% null)" for t, c, p in high_null_cols[:10]])
            chunks.append(Chunk(
                chunk_id=self._make_id("xref"),
                content=f"Columns with high null percentages (>20%): {null_text}.",
                chunk_type="cross_table",
                keywords=["null", "missing", "empty", "incomplete", "data quality"],
                importance_score=1.0
            ))
        
        return chunks

    def _create_stat_chunks(self, results: Dict, db_name: str) -> List[Chunk]:
        """Create statistical summary chunks."""
        chunks = []
        summary = results.get('summary', {})
        
        if summary:
            chunks.append(Chunk(
                chunk_id=self._make_id("stat"),
                content=(
                    f"Database statistics: {summary.get('total_tables', 0)} tables, "
                    f"{summary.get('total_columns', 0)} columns, "
                    f"{summary.get('total_rows', 0):,} total rows, "
                    f"{summary.get('total_relationships', 0)} relationships, "
                    f"quality score {summary.get('quality_score', 0)}/100."
                ),
                chunk_type="stat",
                metadata=summary,
                keywords=["statistics", "count", "total", "numbers", "how many"],
                importance_score=1.5
            ))
        
        return chunks


# ============================================================================
# BM25 SCORER - Keyword relevance scoring
# ============================================================================

class BM25Scorer:
    """
    BM25 (Best Matching 25) scoring for keyword-based relevance.
    
    BM25 formula:
    score(D, Q) = Σ IDF(qi) · (f(qi, D) · (k1 + 1)) / (f(qi, D) + k1 · (1 - b + b · |D|/avgdl))
    
    Parameters:
        k1: Term frequency saturation parameter (default: 1.5)
        b: Length normalization parameter (default: 0.75)
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avg_doc_len = 0
        self.doc_lengths = []
        self.term_doc_freq = defaultdict(int)  # term -> number of docs containing it
        self.doc_term_freq = []  # per-doc term frequencies
        self.vocabulary = set()
        self._fitted = False
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9_\s]', ' ', text)
        tokens = text.split()
        # Add bigrams
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
        return tokens + bigrams
    
    def fit(self, documents: List[str]):
        """Build BM25 index from documents."""
        self.doc_count = len(documents)
        self.doc_term_freq = []
        self.doc_lengths = []
        self.term_doc_freq = defaultdict(int)
        
        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))
            
            term_freq = Counter(tokens)
            self.doc_term_freq.append(term_freq)
            
            for term in set(tokens):
                self.term_doc_freq[term] += 1
                self.vocabulary.add(term)
        
        self.avg_doc_len = sum(self.doc_lengths) / max(self.doc_count, 1)
        self._fitted = True
    
    def _idf(self, term: str) -> float:
        """Calculate Inverse Document Frequency."""
        df = self.term_doc_freq.get(term, 0)
        if df == 0:
            return 0
        return math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
    
    def score(self, query: str, doc_idx: int) -> float:
        """Score a single document against a query."""
        if not self._fitted or doc_idx >= len(self.doc_term_freq):
            return 0.0
        
        query_terms = self._tokenize(query)
        doc_tf = self.doc_term_freq[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        
        score = 0.0
        for term in query_terms:
            tf = doc_tf.get(term, 0)
            idf = self._idf(term)
            
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_len, 1))
            
            score += idf * (numerator / max(denominator, 0.001))
        
        return score
    
    def score_all(self, query: str) -> np.ndarray:
        """Score all documents against a query."""
        scores = np.array([self.score(query, i) for i in range(self.doc_count)])
        return scores


# ============================================================================
# VECTOR STORE - Embedding storage and similarity search
# ============================================================================

class VectorStore:
    """
    Vector store for similarity search with multiple strategies.
    
    Supports:
    1. TF-IDF vectorization (sklearn)
    2. Sentence Transformers (deep learning embeddings)
    3. BM25 keyword scoring
    4. Hybrid search (combination of above)
    
    Index Structure:
    ┌──────────────────────────────┐
    │  TF-IDF Matrix  (sparse)    │
    │  BM25 Index     (inverted)  │
    │  Chunk Store    (metadata)  │
    │  Keyword Index  (inverted)  │
    └──────────────────────────────┘
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else None
        self.chunks: List[Chunk] = []
        self.chunk_texts: List[str] = []
        
        # TF-IDF components
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # BM25 components
        self.bm25_scorer = BM25Scorer()
        
        # Keyword inverted index
        self.keyword_index: Dict[str, Set[int]] = defaultdict(set)
        
        # Sentence transformer (optional)
        self.sentence_model = None
        self.sentence_embeddings = None
        
        self._indexed = False
    
    def add_chunks(self, chunks: List[Chunk]):
        """Add chunks to the store."""
        self.chunks.extend(chunks)
        self.chunk_texts.extend([c.content for c in chunks])
        self._indexed = False
    
    def build_index(self):
        """Build all search indices."""
        if not self.chunks:
            return
        
        print(f"[VectorStore] Building index for {len(self.chunks)} chunks...")
        
        # 1. Build TF-IDF index
        self._build_tfidf_index()
        
        # 2. Build BM25 index
        self._build_bm25_index()
        
        # 3. Build keyword inverted index
        self._build_keyword_index()
        
        # 4. Build sentence embeddings (if available)
        self._build_sentence_index()
        
        self._indexed = True
        print(f"[VectorStore] Index built successfully!")
    
    def _build_tfidf_index(self):
        """Build TF-IDF vectorization index."""
        if not SKLEARN_AVAILABLE:
            return
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),        # Unigrams, bigrams, trigrams
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,         # Apply log normalization
            strip_accents='unicode',
            token_pattern=r'(?u)\b\w[\w.]+\b',  # Include dots for table.column
            stop_words='english'
        )
        
        # Combine content with keywords for richer representation
        enriched_texts = []
        for chunk in self.chunks:
            keywords_text = " ".join(chunk.keywords * 2)  # Boost keywords
            enriched = f"{chunk.content} {keywords_text}"
            enriched_texts.append(enriched)
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(enriched_texts)
        print(f"[VectorStore] TF-IDF index: {self.tfidf_matrix.shape[0]} docs, "
              f"{self.tfidf_matrix.shape[1]} features")
    
    def _build_bm25_index(self):
        """Build BM25 scoring index."""
        self.bm25_scorer = BM25Scorer(k1=1.5, b=0.75)
        # Include keywords in BM25 corpus
        docs = [f"{c.content} {' '.join(c.keywords)}" for c in self.chunks]
        self.bm25_scorer.fit(docs)
        print(f"[VectorStore] BM25 index: {len(docs)} documents, "
              f"{len(self.bm25_scorer.vocabulary)} terms")
    
    def _build_keyword_index(self):
        """Build inverted keyword index for fast lookup."""
        self.keyword_index = defaultdict(set)
        
        for idx, chunk in enumerate(self.chunks):
            for keyword in chunk.keywords:
                self.keyword_index[keyword.lower()].add(idx)
            
            # Also index words from content
            words = set(re.findall(r'\b[a-z_]{3,}\b', chunk.content.lower()))
            for word in words:
                self.keyword_index[word].add(idx)
        
        print(f"[VectorStore] Keyword index: {len(self.keyword_index)} unique terms")
    
    def _build_sentence_index(self):
        """Build sentence transformer embeddings if available."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return
        
        try:
            if self.sentence_model is None:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            texts = [c.content for c in self.chunks]
            self.sentence_embeddings = self.sentence_model.encode(texts, show_progress_bar=False)
            print(f"[VectorStore] Sentence embeddings: {self.sentence_embeddings.shape}")
        except Exception as e:
            print(f"[VectorStore] Sentence transformer unavailable: {e}")
    
    def search(self, query: str, top_k: int = 10, strategy: str = "hybrid") -> List[SearchResult]:
        """
        Search for relevant chunks.
        
        Strategies:
        - 'tfidf': TF-IDF cosine similarity
        - 'bm25': BM25 keyword scoring
        - 'keyword': Exact keyword matching
        - 'semantic': Sentence transformer similarity
        - 'hybrid': Weighted combination (default)
        """
        if not self._indexed or not self.chunks:
            return []
        
        if strategy == "tfidf":
            return self._search_tfidf(query, top_k)
        elif strategy == "bm25":
            return self._search_bm25(query, top_k)
        elif strategy == "keyword":
            return self._search_keyword(query, top_k)
        elif strategy == "semantic" and SENTENCE_TRANSFORMERS_AVAILABLE:
            return self._search_semantic(query, top_k)
        else:
            return self._search_hybrid(query, top_k)
    
    def _search_tfidf(self, query: str, top_k: int) -> List[SearchResult]:
        """TF-IDF cosine similarity search."""
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            return []
        
        query_vec = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Apply importance weighting
        for i, chunk in enumerate(self.chunks):
            similarities[i] *= chunk.importance_score
        
        top_indices = similarities.argsort()[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            if similarities[idx] > 0.01:
                results.append(SearchResult(
                    chunk=self.chunks[idx],
                    score=float(similarities[idx]),
                    rank=rank + 1,
                    match_type="tfidf"
                ))
        
        return results
    
    def _search_bm25(self, query: str, top_k: int) -> List[SearchResult]:
        """BM25 keyword relevance search."""
        scores = self.bm25_scorer.score_all(query)
        
        # Apply importance weighting
        for i, chunk in enumerate(self.chunks):
            scores[i] *= chunk.importance_score
        
        top_indices = scores.argsort()[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0.01:
                results.append(SearchResult(
                    chunk=self.chunks[idx],
                    score=float(scores[idx]),
                    rank=rank + 1,
                    match_type="bm25"
                ))
        
        return results
    
    def _search_keyword(self, query: str, top_k: int) -> List[SearchResult]:
        """Exact keyword matching search."""
        query_terms = set(re.findall(r'\b[a-z_]{2,}\b', query.lower()))
        
        # Score by number of matching keywords
        scores = np.zeros(len(self.chunks))
        for term in query_terms:
            matching_indices = self.keyword_index.get(term, set())
            for idx in matching_indices:
                scores[idx] += 1
        
        # Normalize by number of query terms
        if query_terms:
            scores /= len(query_terms)
        
        # Apply importance
        for i, chunk in enumerate(self.chunks):
            scores[i] *= chunk.importance_score
        
        top_indices = scores.argsort()[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:
                results.append(SearchResult(
                    chunk=self.chunks[idx],
                    score=float(scores[idx]),
                    rank=rank + 1,
                    match_type="keyword"
                ))
        
        return results
    
    def _search_semantic(self, query: str, top_k: int) -> List[SearchResult]:
        """Sentence transformer semantic search."""
        if self.sentence_model is None or self.sentence_embeddings is None:
            return self._search_tfidf(query, top_k)
        
        query_embedding = self.sentence_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.sentence_embeddings).flatten()
        
        for i, chunk in enumerate(self.chunks):
            similarities[i] *= chunk.importance_score
        
        top_indices = similarities.argsort()[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            if similarities[idx] > 0.1:
                results.append(SearchResult(
                    chunk=self.chunks[idx],
                    score=float(similarities[idx]),
                    rank=rank + 1,
                    match_type="semantic"
                ))
        
        return results
    
    def _search_hybrid(self, query: str, top_k: int) -> List[SearchResult]:
        """
        Hybrid search combining TF-IDF + BM25 + keyword matching.
        
        Weights:
        - TF-IDF: 0.4 (semantic-ish similarity)
        - BM25: 0.35 (keyword relevance)
        - Keyword: 0.25 (exact match boost)
        """
        n = len(self.chunks)
        if n == 0:
            return []
        
        final_scores = np.zeros(n)
        
        # TF-IDF scores (normalized)
        if self.tfidf_vectorizer and self.tfidf_matrix is not None:
            query_vec = self.tfidf_vectorizer.transform([query])
            tfidf_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            max_tfidf = tfidf_scores.max() if tfidf_scores.max() > 0 else 1
            final_scores += 0.4 * (tfidf_scores / max_tfidf)
        
        # BM25 scores (normalized)
        bm25_scores = self.bm25_scorer.score_all(query)
        max_bm25 = bm25_scores.max() if bm25_scores.max() > 0 else 1
        final_scores += 0.35 * (bm25_scores / max_bm25)
        
        # Keyword scores (normalized)
        query_terms = set(re.findall(r'\b[a-z_]{2,}\b', query.lower()))
        keyword_scores = np.zeros(n)
        for term in query_terms:
            for idx in self.keyword_index.get(term, set()):
                keyword_scores[idx] += 1
        max_kw = keyword_scores.max() if keyword_scores.max() > 0 else 1
        final_scores += 0.25 * (keyword_scores / max_kw)
        
        # Apply importance weighting
        for i, chunk in enumerate(self.chunks):
            final_scores[i] *= chunk.importance_score
        
        # Apply MMR (Maximal Marginal Relevance) for diversity
        results = self._mmr_rerank(final_scores, top_k, lambda_param=0.7)
        
        return results
    
    def _mmr_rerank(self, scores: np.ndarray, top_k: int, lambda_param: float = 0.7) -> List[SearchResult]:
        """
        Maximal Marginal Relevance re-ranking for diversity.
        
        MMR = λ · Sim(qi, di) - (1-λ) · max(Sim(di, dj)) for dj in Selected
        
        This ensures retrieved chunks are both relevant AND diverse.
        """
        n = len(scores)
        if n == 0:
            return []
        
        selected = []
        remaining = set(range(n))
        
        # Get TF-IDF vectors for diversity calculation
        if self.tfidf_matrix is not None:
            doc_sim_matrix = cosine_similarity(self.tfidf_matrix)
        else:
            doc_sim_matrix = np.eye(n)
        
        for _ in range(min(top_k, n)):
            if not remaining:
                break
            
            best_idx = -1
            best_mmr = -float('inf')
            
            for idx in remaining:
                if scores[idx] < 0.01:
                    continue
                
                relevance = scores[idx]
                
                # Max similarity to already selected docs
                max_sim = 0
                for sel_idx in selected:
                    sim = doc_sim_matrix[idx][sel_idx] if self.tfidf_matrix is not None else 0
                    max_sim = max(max_sim, sim)
                
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx
            
            if best_idx == -1:
                break
            
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        results = []
        for rank, idx in enumerate(selected):
            results.append(SearchResult(
                chunk=self.chunks[idx],
                score=float(scores[idx]),
                rank=rank + 1,
                match_type="hybrid"
            ))
        
        return results
    
    def save_index(self, dataset_id: str):
        """Persist the vector index to disk."""
        if not self.storage_path:
            return
        
        index_dir = self.storage_path / 'outputs' / dataset_id / 'vector_index'
        index_dir.mkdir(parents=True, exist_ok=True)
        
        # Save chunks
        chunks_data = [c.to_dict() for c in self.chunks]
        with open(index_dir / 'chunks.json', 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2, default=str)
        
        # Save TF-IDF model
        if self.tfidf_vectorizer and self.tfidf_matrix is not None:
            with open(index_dir / 'tfidf_vectorizer.pkl', 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
            
            from scipy import sparse
            sparse.save_npz(index_dir / 'tfidf_matrix.npz', self.tfidf_matrix)
        
        # Save BM25
        with open(index_dir / 'bm25.pkl', 'wb') as f:
            pickle.dump(self.bm25_scorer, f)
        
        # Save metadata
        meta = {
            'dataset_id': dataset_id,
            'chunk_count': len(self.chunks),
            'created_at': datetime.now().isoformat(),
            'strategies': ['tfidf', 'bm25', 'keyword'] + 
                         (['semantic'] if self.sentence_embeddings is not None else []),
            'tfidf_features': self.tfidf_matrix.shape[1] if self.tfidf_matrix is not None else 0,
            'bm25_vocab_size': len(self.bm25_scorer.vocabulary)
        }
        with open(index_dir / 'index_meta.json', 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
        
        print(f"[VectorStore] Index saved for dataset {dataset_id[:16]}...")
    
    def load_index(self, dataset_id: str) -> bool:
        """Load persisted vector index from disk."""
        if not self.storage_path:
            return False
        
        index_dir = self.storage_path / 'outputs' / dataset_id / 'vector_index'
        
        if not (index_dir / 'chunks.json').exists():
            return False
        
        try:
            # Load chunks
            with open(index_dir / 'chunks.json', 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            self.chunks = [Chunk.from_dict(d) for d in chunks_data]
            self.chunk_texts = [c.content for c in self.chunks]
            
            # Load TF-IDF
            tfidf_path = index_dir / 'tfidf_vectorizer.pkl'
            matrix_path = index_dir / 'tfidf_matrix.npz'
            if tfidf_path.exists() and matrix_path.exists():
                with open(tfidf_path, 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
                from scipy import sparse
                self.tfidf_matrix = sparse.load_npz(matrix_path)
            
            # Load BM25
            bm25_path = index_dir / 'bm25.pkl'
            if bm25_path.exists():
                with open(bm25_path, 'rb') as f:
                    self.bm25_scorer = pickle.load(f)
            
            # Rebuild keyword index
            self._build_keyword_index()
            
            self._indexed = True
            print(f"[VectorStore] Index loaded: {len(self.chunks)} chunks")
            return True
        except Exception as e:
            print(f"[VectorStore] Failed to load index: {e}")
            return False


# ============================================================================
# RESPONSE GENERATOR - Synthesizes answers from retrieved chunks
# ============================================================================

class ResponseGenerator:
    """
    Generates human-readable answers from retrieved chunks.
    
    Uses template-based generation with intelligent context assembly.
    """
    
    QUERY_PATTERNS = {
        'table_info': [
            r'\b(table|tables)\b', r'\bschema\b', r'\bstructure\b',
            r'\bwhat\s+(is|are)\s+the\s+table', r'\bhow\s+many\s+table'
        ],
        'column_info': [
            r'\bcolumn', r'\bfield', r'\bdata\s*type', r'\bnullable',
            r'\bprimary\s+key', r'\bforeign\s+key'
        ],
        'relationship': [
            r'\brelation', r'\bjoin', r'\bconnect', r'\blink',
            r'\breference', r'\bforeign\s+key'
        ],
        'quality': [
            r'\bquality', r'\bissue', r'\bproblem', r'\bnull',
            r'\bmissing', r'\bincomplete', r'\berror'
        ],
        'query_help': [
            r'\bquery', r'\bsql', r'\bselect', r'\bhow\s+to\s+get',
            r'\bhow\s+to\s+find', r'\bshow\s+me'
        ],
        'stats': [
            r'\bhow\s+many', r'\bcount', r'\btotal', r'\bsize',
            r'\brows?\b', r'\brecord'
        ]
    }
    
    def generate_response(self, query: str, results: List[SearchResult]) -> str:
        """Generate a comprehensive answer from search results."""
        if not results:
            return "I couldn't find relevant information for your query. Try rephrasing or ask about specific tables, columns, relationships, or data quality."
        
        # Detect query intent
        intent = self._detect_intent(query)
        
        # Build context from top results
        context_parts = []
        seen_content = set()
        
        for result in results[:7]:  # Use top 7 chunks
            content = result.chunk.content.strip()
            # Deduplicate
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                context_parts.append({
                    'content': content,
                    'type': result.chunk.chunk_type,
                    'score': result.score,
                    'table': result.chunk.source_table,
                    'column': result.chunk.source_column
                })
        
        # Assemble answer
        answer = self._assemble_answer(query, intent, context_parts)
        
        return answer
    
    def _detect_intent(self, query: str) -> str:
        """Detect the intent of the query."""
        query_lower = query.lower()
        
        best_intent = 'general'
        best_score = 0
        
        for intent, patterns in self.QUERY_PATTERNS.items():
            score = sum(1 for p in patterns if re.search(p, query_lower))
            if score > best_score:
                best_score = score
                best_intent = intent
        
        return best_intent
    
    def _assemble_answer(self, query: str, intent: str, context_parts: List[Dict]) -> str:
        """Assemble a structured answer."""
        if not context_parts:
            return "No relevant information found."
        
        parts = []
        
        # Primary answer from highest-scored chunk
        primary = context_parts[0]
        parts.append(f"**{primary['content']}**")
        
        # Add supporting details
        if len(context_parts) > 1:
            parts.append("\n\n**Additional Details:**")
            for ctx in context_parts[1:5]:
                prefix = self._get_chunk_prefix(ctx['type'])
                parts.append(f"\n- {prefix}{ctx['content']}")
        
        # Add SQL suggestion if query intent
        if intent == 'query_help':
            sql_chunks = [c for c in context_parts if c['type'] == 'query']
            if sql_chunks:
                parts.append(f"\n\n**Suggested SQL:**\n```sql\n{sql_chunks[0]['content'].split('SQL: ')[-1] if 'SQL: ' in sql_chunks[0]['content'] else ''}\n```")
        
        return "\n".join(parts)
    
    def _get_chunk_prefix(self, chunk_type: str) -> str:
        prefixes = {
            'table_overview': '📊 ',
            'column_detail': '📋 ',
            'relationship': '🔗 ',
            'quality': '⚠️ ',
            'query': '💡 ',
            'stat': '📈 ',
            'cross_table': '🔀 ',
            'db_overview': '🗄️ '
        }
        return prefixes.get(chunk_type, '')


# ============================================================================
# RAG ENGINE - Main orchestrator
# ============================================================================

class RAGEngine:
    """
    Main RAG (Retrieval-Augmented Generation) engine.
    
    Orchestrates the full pipeline:
    Query → Vectorize → Search → Rerank → Generate Response
    
    Usage:
        rag = RAGEngine(storage_path="/path/to/storage")
        rag.index_dataset(dataset_id, analysis_results)
        response = rag.query(dataset_id, "How many tables are there?")
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self.chunker = SchemaChunker()
        self.response_generator = ResponseGenerator()
        self._stores: Dict[str, VectorStore] = {}  # dataset_id -> VectorStore
    
    def index_dataset(self, dataset_id: str, analysis_results: Dict) -> Dict:
        """
        Index a dataset's analysis results for search.
        
        Steps:
        1. Chunk the analysis results
        2. Build vector indices (TF-IDF + BM25 + keywords)
        3. Persist index to disk
        
        Returns: Indexing statistics
        """
        import time
        start = time.time()
        
        # Check for cached index
        store = VectorStore(storage_path=self.storage_path)
        if store.load_index(dataset_id):
            self._stores[dataset_id] = store
            return {
                'status': 'loaded_from_cache',
                'chunk_count': len(store.chunks),
                'message': 'Vector index loaded from cache'
            }
        
        # Chunk the results
        chunks = self.chunker.chunk_analysis(analysis_results)
        
        # Build vector store
        store = VectorStore(storage_path=self.storage_path)
        store.add_chunks(chunks)
        store.build_index()
        
        # Save to disk
        store.save_index(dataset_id)
        
        # Cache in memory
        self._stores[dataset_id] = store
        
        elapsed = (time.time() - start) * 1000
        
        stats = {
            'status': 'indexed',
            'chunk_count': len(chunks),
            'chunk_types': dict(Counter(c.chunk_type for c in chunks)),
            'indexing_time_ms': round(elapsed, 2),
            'strategies': ['tfidf', 'bm25', 'keyword'],
            'message': f'Indexed {len(chunks)} chunks in {elapsed:.0f}ms'
        }
        
        print(f"[RAG] Dataset {dataset_id[:16]}... indexed: "
              f"{len(chunks)} chunks in {elapsed:.0f}ms")
        
        return stats
    
    def query(self, dataset_id: str, query_text: str, 
              top_k: int = 10, strategy: str = "hybrid") -> QueryResponse:
        """
        Answer a query about a dataset.
        
        Args:
            dataset_id: Dataset identifier
            query_text: User's natural language query
            top_k: Number of chunks to retrieve
            strategy: Search strategy (hybrid, tfidf, bm25, keyword, semantic)
        
        Returns:
            QueryResponse with answer and supporting evidence
        """
        import time
        start = time.time()
        
        # Get or load vector store
        store = self._get_store(dataset_id)
        if store is None:
            return QueryResponse(
                query=query_text,
                answer="Dataset not indexed. Please analyze the database first.",
                relevant_chunks=[],
                confidence=0,
                processing_time_ms=0,
                total_chunks_searched=0
            )
        
        # Search for relevant chunks
        results = store.search(query_text, top_k=top_k, strategy=strategy)
        
        # Generate response
        answer = self.response_generator.generate_response(query_text, results)
        
        # Calculate confidence
        confidence = self._calculate_confidence(results)
        
        elapsed = (time.time() - start) * 1000
        
        return QueryResponse(
            query=query_text,
            answer=answer,
            relevant_chunks=results,
            confidence=confidence,
            processing_time_ms=round(elapsed, 2),
            total_chunks_searched=len(store.chunks)
        )
    
    def _get_store(self, dataset_id: str) -> Optional[VectorStore]:
        """Get vector store, loading from disk if needed."""
        if dataset_id in self._stores:
            return self._stores[dataset_id]
        
        # Try loading from disk
        store = VectorStore(storage_path=self.storage_path)
        if store.load_index(dataset_id):
            self._stores[dataset_id] = store
            return store
        
        return None
    
    def _calculate_confidence(self, results: List[SearchResult]) -> float:
        """Calculate confidence score for the response."""
        if not results:
            return 0.0
        
        # Based on top scores and number of relevant chunks
        top_score = results[0].score if results else 0
        avg_score = np.mean([r.score for r in results[:5]]) if results else 0
        coverage = min(len(results) / 5.0, 1.0)  # Having 5+ results = full coverage
        
        confidence = (top_score * 0.4 + avg_score * 0.3 + coverage * 0.3)
        return round(min(confidence, 1.0), 3)
    
    def get_index_stats(self, dataset_id: str) -> Optional[Dict]:
        """Get statistics about a dataset's vector index."""
        store = self._get_store(dataset_id)
        if store is None:
            return None
        
        chunk_types = Counter(c.chunk_type for c in store.chunks)
        
        return {
            'dataset_id': dataset_id,
            'total_chunks': len(store.chunks),
            'chunk_types': dict(chunk_types),
            'indexed': store._indexed,
            'has_tfidf': store.tfidf_matrix is not None,
            'has_bm25': store.bm25_scorer._fitted,
            'has_semantic': store.sentence_embeddings is not None,
            'tfidf_features': store.tfidf_matrix.shape[1] if store.tfidf_matrix is not None else 0,
            'bm25_vocab': len(store.bm25_scorer.vocabulary),
        }


# ============================================================================
# FACTORY
# ============================================================================

def create_rag_engine(storage_path: str) -> RAGEngine:
    """Create a RAG engine instance."""
    return RAGEngine(storage_path=storage_path)
