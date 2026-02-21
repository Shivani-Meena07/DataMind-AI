"""
Prompt Templates for LLM Inference

Contains all prompts used for semantic analysis and documentation generation.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class PromptTemplates:
    """Collection of prompt templates for various inference tasks."""
    
    # System prompt establishing the AI's role
    SYSTEM_PROMPT = """You are an expert database analyst, business analyst, and technical writer.

Your task is to analyze database schemas and data patterns to generate clear, accurate, and insightful documentation.

Guidelines:
1. Base all explanations ONLY on the provided schema and data information
2. Use clear, professional language suitable for both technical and non-technical audiences
3. If making an inference that isn't directly evident from the data, clearly mark it as "Inferred:" or "Assumed:"
4. Focus on business value and practical usage
5. Be specific and avoid generic descriptions
6. Consider the context of e-commerce, logistics, and customer management when relevant

Never hallucinate or make up information not supported by the provided data."""

    # Template for analyzing a single table
    TABLE_ANALYSIS_PROMPT = """Analyze the following database table and provide comprehensive business documentation.

TABLE: {table_name}
SCHEMA: {schema}
ROW COUNT: {row_count}
TABLE TYPE: {table_type}

COLUMNS:
{columns_info}

PRIMARY KEY: {primary_key}
FOREIGN KEYS: {foreign_keys}

INCOMING RELATIONSHIPS (tables that reference this table):
{incoming_relationships}

OUTGOING RELATIONSHIPS (tables this table references):
{outgoing_relationships}

DATA QUALITY METRICS:
{quality_metrics}

Based on this information, provide:

1. BUSINESS ENTITY: What real-world business entity does this table represent? (1-2 sentences)

2. PURPOSE: Why does this table exist in the database? What business need does it serve? (2-3 sentences)

3. PRIMARY USERS: Which business teams or systems would primarily use this table? (list 2-4 users)

4. UPDATE FREQUENCY: Based on the table type and data patterns, how often is data in this table likely updated?
   (Options: Real-time, Hourly, Daily, Weekly, Rarely/Static)

5. KEY INSIGHTS: What are 2-3 important things to understand about this table?

Respond in JSON format:
{{
    "business_entity": "...",
    "purpose": "...",
    "primary_users": ["...", "..."],
    "update_frequency": "...",
    "key_insights": ["...", "...", "..."]
}}"""

    # Template for analyzing columns in batch
    COLUMN_ANALYSIS_PROMPT = """Analyze the following columns from the table '{table_name}' and provide business descriptions.

TABLE CONTEXT: {table_context}

COLUMNS TO ANALYZE:
{columns_info}

For each column, provide:
1. A plain English description of what this column represents (1-2 sentences)
2. Its business relevance (why is this column important?)
3. Common usage scenarios in reporting/analytics

Respond in JSON format with column names as keys:
{{
    "column_name": {{
        "description": "Plain English meaning...",
        "business_relevance": "Why this matters...",
        "common_usage": "How this is typically used in analytics..."
    }},
    ...
}}"""

    # Template for relationship analysis
    RELATIONSHIP_ANALYSIS_PROMPT = """Analyze the following database relationship and explain it in business terms.

SOURCE TABLE: {source_table}
SOURCE COLUMNS: {source_columns}
TARGET TABLE: {target_table}  
TARGET COLUMNS: {target_columns}
CARDINALITY: {cardinality}

SOURCE TABLE CONTEXT:
{source_context}

TARGET TABLE CONTEXT:
{target_context}

Provide:
1. A business-friendly explanation of this relationship (2-3 sentences)
2. How data flows between these tables
3. A practical example of how this relationship is used

Respond in JSON format:
{{
    "business_description": "...",
    "data_flow": "...",
    "example_usage": "..."
}}"""

    # Template for generating executive summary
    EXECUTIVE_SUMMARY_PROMPT = """Generate an executive summary for the following database.

DATABASE OVERVIEW:
- Database Type: {db_type}
- Total Tables: {total_tables}
- Total Rows: {total_rows}
- Total Relationships: {total_relationships}

TABLE SUMMARY:
{table_summary}

KEY RELATIONSHIPS:
{relationship_summary}

DATA QUALITY OVERVIEW:
- Overall Quality Score: {quality_score}/100
- Critical Issues: {critical_issues}
- Warning Issues: {warning_issues}

Generate a 3-4 paragraph executive summary that:
1. Describes what this database is used for (based on the tables and relationships)
2. Highlights the key business domains covered
3. Mentions any significant data quality concerns
4. Provides a high-level assessment suitable for non-technical stakeholders

Write in clear, professional prose without bullet points or technical jargon."""

    # Template for generating business use cases
    USE_CASE_INFERENCE_PROMPT = """Based on the following database structure, infer the key business use cases and analytics questions this database can answer.

DATABASE STRUCTURE:
{schema_summary}

KEY TABLES AND THEIR PURPOSES:
{table_purposes}

RELATIONSHIPS:
{relationships}

Generate:
1. 5-7 key business questions this database can answer
2. 3-4 suggested KPIs that can be derived from this data
3. Recommended analytics use cases organized by business domain

Respond in JSON format:
{{
    "business_questions": [
        {{"question": "...", "relevant_tables": ["...", "..."], "complexity": "simple/moderate/complex"}},
        ...
    ],
    "suggested_kpis": [
        {{"name": "...", "description": "...", "formula_hint": "...", "relevant_tables": ["..."]}},
        ...
    ],
    "analytics_use_cases": [
        {{"domain": "...", "use_case": "...", "description": "...", "tables_involved": ["..."]}},
        ...
    ]
}}"""

    # Template for generating sample queries
    SAMPLE_QUERY_PROMPT = """Generate sample SQL queries for the following business questions using this database schema.

DATABASE TYPE: {db_type}
SCHEMA:
{schema_info}

BUSINESS QUESTIONS:
{questions}

For each question, provide:
1. A well-commented SQL query
2. Brief explanation of what the query does
3. Any caveats or assumptions

Respond in JSON format:
{{
    "queries": [
        {{
            "question": "...",
            "sql": "...",
            "explanation": "...",
            "caveats": "..."
        }},
        ...
    ]
}}"""

    # Template for data quality recommendations
    QUALITY_RECOMMENDATIONS_PROMPT = """Analyze the following data quality issues and provide recommendations.

DATA QUALITY ISSUES:
{quality_issues}

TABLE CONTEXT:
{table_context}

For each category of issues, provide:
1. Root cause analysis (what might be causing these issues?)
2. Business impact (how do these issues affect data consumers?)
3. Recommended remediation steps
4. Prevention strategies

Respond in JSON format:
{{
    "analysis": [
        {{
            "issue_category": "...",
            "root_cause": "...",
            "business_impact": "...",
            "remediation": ["...", "..."],
            "prevention": ["...", "..."]
        }},
        ...
    ],
    "priority_actions": ["...", "...", "..."]
}}"""

    # Template for improvement suggestions
    IMPROVEMENT_SUGGESTIONS_PROMPT = """Based on the following database analysis, suggest improvements.

SCHEMA SUMMARY:
{schema_summary}

IDENTIFIED ISSUES:
{issues}

CURRENT PATTERNS:
{patterns}

Provide:
1. Schema improvement suggestions (normalization, denormalization, indexing)
2. Data governance recommendations
3. Documentation recommendations
4. Performance optimization hints

Respond in JSON format:
{{
    "schema_improvements": [
        {{"suggestion": "...", "rationale": "...", "priority": "high/medium/low", "effort": "low/medium/high"}}
    ],
    "governance_recommendations": ["..."],
    "documentation_recommendations": ["..."],
    "performance_hints": ["..."]
}}"""

    @classmethod
    def format_table_analysis(
        cls,
        table_name: str,
        schema: Optional[str],
        row_count: int,
        table_type: str,
        columns: List[Dict],
        primary_key: Optional[List[str]],
        foreign_keys: List[Dict],
        incoming_rels: List[Dict],
        outgoing_rels: List[Dict],
        quality_metrics: Dict
    ) -> str:
        """Format the table analysis prompt with provided data."""
        
        # Format columns info
        columns_info = "\n".join([
            f"- {c['name']} ({c['data_type']}): "
            f"NULL rate: {c.get('null_pct', 0):.1%}, "
            f"Unique: {c.get('distinct_count', 0)}, "
            f"Type: {c.get('semantic_type', 'unknown')}"
            f"{' [PK]' if c.get('is_pk') else ''}"
            f"{' [FK -> ' + c.get('ref_table', '') + ']' if c.get('is_fk') else ''}"
            for c in columns
        ])
        
        # Format relationships
        incoming_str = "\n".join([
            f"- {r['table']}.{r['column']} -> {table_name}"
            for r in incoming_rels
        ]) or "None"
        
        outgoing_str = "\n".join([
            f"- {table_name} -> {r['table']}.{r['column']}"
            for r in outgoing_rels
        ]) or "None"
        
        # Format quality metrics
        quality_str = "\n".join([
            f"- {k}: {v}" for k, v in quality_metrics.items()
        ])
        
        return cls.TABLE_ANALYSIS_PROMPT.format(
            table_name=table_name,
            schema=schema or "default",
            row_count=row_count,
            table_type=table_type,
            columns_info=columns_info,
            primary_key=", ".join(primary_key) if primary_key else "None",
            foreign_keys=str(foreign_keys) if foreign_keys else "None",
            incoming_relationships=incoming_str,
            outgoing_relationships=outgoing_str,
            quality_metrics=quality_str,
        )
    
    @classmethod
    def format_column_analysis(
        cls,
        table_name: str,
        table_context: str,
        columns: List[Dict]
    ) -> str:
        """Format the column analysis prompt."""
        columns_info = "\n\n".join([
            f"Column: {c['name']}\n"
            f"  Data Type: {c['data_type']}\n"
            f"  Nullable: {c['nullable']}\n"
            f"  Null Rate: {c.get('null_pct', 0):.1%}\n"
            f"  Distinct Values: {c.get('distinct_count', 'N/A')}\n"
            f"  Sample Values: {c.get('samples', [])[:5]}\n"
            f"  Is Primary Key: {c.get('is_pk', False)}\n"
            f"  Is Foreign Key: {c.get('is_fk', False)}"
            for c in columns
        ])
        
        return cls.COLUMN_ANALYSIS_PROMPT.format(
            table_name=table_name,
            table_context=table_context,
            columns_info=columns_info
        )
    
    @classmethod 
    def format_executive_summary(
        cls,
        db_type: str,
        total_tables: int,
        total_rows: int,
        total_relationships: int,
        table_summary: str,
        relationship_summary: str,
        quality_score: float,
        critical_issues: int,
        warning_issues: int
    ) -> str:
        """Format the executive summary prompt."""
        return cls.EXECUTIVE_SUMMARY_PROMPT.format(
            db_type=db_type,
            total_tables=total_tables,
            total_rows=total_rows,
            total_relationships=total_relationships,
            table_summary=table_summary,
            relationship_summary=relationship_summary,
            quality_score=quality_score,
            critical_issues=critical_issues,
            warning_issues=warning_issues
        )
