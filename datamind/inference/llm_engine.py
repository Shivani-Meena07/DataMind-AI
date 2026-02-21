"""
LLM Inference Engine

Coordinates LLM calls to generate semantic understanding
of database structures, relationships, and business context.
"""

import json
import logging
from typing import Dict, List, Optional, Any
import time

from datamind.core.intelligence_store import (
    IntelligenceStore,
    TableProfile,
    ColumnProfile,
    Relationship,
    BusinessInsight,
)
from datamind.inference.prompts import PromptTemplates
from datamind.config import DataMindConfig, LLMConfig

logger = logging.getLogger(__name__)


class LLMInferenceEngine:
    """
    LLM-based inference engine for semantic database analysis.
    
    This module handles:
    - Table business entity classification
    - Column meaning inference
    - Relationship interpretation
    - Use case generation
    - Executive summary generation
    - Data quality recommendation generation
    """
    
    def __init__(self, config: DataMindConfig):
        """
        Initialize the LLM inference engine.
        
        Args:
            config: DataMind configuration
        """
        self.config = config
        self.llm_config = config.llm
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the LLM client."""
        if self.llm_config.provider == "openai":
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.llm_config.api_key)
                logger.info(f"Initialized OpenAI client with model: {self.llm_config.model}")
            except ImportError:
                logger.warning("OpenAI package not installed. LLM features will be disabled.")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        else:
            logger.warning(f"Unsupported LLM provider: {self.llm_config.provider}")
    
    def infer(self, store: IntelligenceStore) -> IntelligenceStore:
        """
        Run LLM inference on the intelligence store.
        
        Args:
            store: Intelligence store with schema and profiling data
            
        Returns:
            Intelligence store with LLM-generated insights
        """
        if not self._client:
            logger.warning("LLM client not available. Skipping inference.")
            return store
        
        logger.info("Starting LLM inference...")
        
        # Analyze tables
        self._analyze_tables(store)
        
        # Analyze columns
        self._analyze_columns(store)
        
        # Analyze relationships
        self._analyze_relationships(store)
        
        # Generate use cases and insights
        self._generate_use_cases(store)
        
        # Generate executive summary
        self._generate_executive_summary(store)
        
        # Generate improvement suggestions
        self._generate_improvement_suggestions(store)
        
        logger.info("LLM inference complete.")
        return store
    
    def _call_llm(
        self, 
        prompt: str, 
        system_prompt: str = None,
        expect_json: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Make a call to the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt override
            expect_json: Whether to parse response as JSON
            
        Returns:
            Parsed response or None on failure
        """
        if not self._client:
            return None
        
        system = system_prompt or PromptTemplates.SYSTEM_PROMPT
        
        for attempt in range(self.llm_config.retry_attempts):
            try:
                response = self._client.chat.completions.create(
                    model=self.llm_config.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.llm_config.temperature,
                    max_tokens=self.llm_config.max_tokens,
                    response_format={"type": "json_object"} if expect_json else None,
                )
                
                content = response.choices[0].message.content
                
                if expect_json:
                    return json.loads(content)
                return {"text": content}
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {e}")
                if attempt < self.llm_config.retry_attempts - 1:
                    time.sleep(1)
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt < self.llm_config.retry_attempts - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def _analyze_tables(self, store: IntelligenceStore):
        """Analyze each table to infer business meaning."""
        logger.info("Analyzing tables...")
        
        for table_name, table in store.tables.items():
            try:
                result = self._analyze_single_table(table, store)
                if result:
                    table.business_entity = result.get("business_entity")
                    table.business_description = result.get("purpose")
                    table.primary_users = ", ".join(result.get("primary_users", []))
                    table.update_frequency = result.get("update_frequency")
                    
                    # Add key insights
                    for insight in result.get("key_insights", []):
                        store.add_insight(BusinessInsight(
                            category="TableInsight",
                            title=f"About {table_name}",
                            description=insight,
                            related_tables=[table_name],
                        ))
                        
            except Exception as e:
                logger.warning(f"Failed to analyze table {table_name}: {e}")
    
    def _analyze_single_table(
        self, 
        table: TableProfile, 
        store: IntelligenceStore
    ) -> Optional[Dict]:
        """Analyze a single table."""
        # Prepare column info
        columns = []
        for col_name, col in table.columns.items():
            columns.append({
                "name": col_name,
                "data_type": col.data_type,
                "null_pct": col.null_percentage,
                "distinct_count": col.distinct_count,
                "semantic_type": col.semantic_type.value,
                "is_pk": col.is_primary_key,
                "is_fk": col.is_foreign_key,
                "ref_table": col.references_table,
            })
        
        # Get relationships
        incoming = [
            {"table": r.source_table, "column": r.source_columns[0]}
            for r in store.get_incoming_relationships(table.name)
            if r.source_columns
        ]
        
        outgoing = [
            {"table": r.target_table, "column": r.target_columns[0]}
            for r in store.get_outgoing_relationships(table.name)
            if r.target_columns
        ]
        
        # Quality metrics
        quality = {
            "row_count": table.row_count,
            "columns_with_nulls": sum(1 for c in table.columns.values() if c.null_count > 0),
            "high_null_columns": sum(1 for c in table.columns.values() if c.null_percentage > 0.5),
        }
        
        prompt = PromptTemplates.format_table_analysis(
            table_name=table.name,
            schema=table.schema,
            row_count=table.row_count,
            table_type=table.table_type.value,
            columns=columns,
            primary_key=table.primary_key,
            foreign_keys=table.foreign_keys,
            incoming_rels=incoming,
            outgoing_rels=outgoing,
            quality_metrics=quality,
        )
        
        return self._call_llm(prompt)
    
    def _analyze_columns(self, store: IntelligenceStore):
        """Analyze columns to infer business meaning."""
        logger.info("Analyzing columns...")
        
        for table_name, table in store.tables.items():
            try:
                # Process columns in batches
                columns_list = list(table.columns.values())
                batch_size = self.llm_config.batch_size
                
                for i in range(0, len(columns_list), batch_size):
                    batch = columns_list[i:i + batch_size]
                    result = self._analyze_column_batch(table, batch)
                    
                    if result:
                        for col_name, col_info in result.items():
                            if col_name in table.columns:
                                col = table.columns[col_name]
                                col.business_description = col_info.get("description")
                                col.business_relevance = col_info.get("business_relevance")
                                col.common_usage = col_info.get("common_usage")
                                
            except Exception as e:
                logger.warning(f"Failed to analyze columns for {table_name}: {e}")
    
    def _analyze_column_batch(
        self, 
        table: TableProfile, 
        columns: List[ColumnProfile]
    ) -> Optional[Dict]:
        """Analyze a batch of columns."""
        columns_info = []
        for col in columns:
            columns_info.append({
                "name": col.name,
                "data_type": col.data_type,
                "nullable": col.nullable,
                "null_pct": col.null_percentage,
                "distinct_count": col.distinct_count,
                "samples": col.sample_values[:5],
                "is_pk": col.is_primary_key,
                "is_fk": col.is_foreign_key,
            })
        
        table_context = f"{table.name}: {table.business_description or 'No description yet'}"
        
        prompt = PromptTemplates.format_column_analysis(
            table_name=table.name,
            table_context=table_context,
            columns=columns_info,
        )
        
        return self._call_llm(prompt)
    
    def _analyze_relationships(self, store: IntelligenceStore):
        """Analyze relationships to generate business descriptions."""
        logger.info("Analyzing relationships...")
        
        for rel in store.relationships:
            try:
                source_table = store.tables.get(rel.source_table)
                target_table = store.tables.get(rel.target_table)
                
                if not source_table or not target_table:
                    continue
                
                prompt = PromptTemplates.RELATIONSHIP_ANALYSIS_PROMPT.format(
                    source_table=rel.source_table,
                    source_columns=", ".join(rel.source_columns),
                    target_table=rel.target_table,
                    target_columns=", ".join(rel.target_columns),
                    cardinality=rel.cardinality,
                    source_context=source_table.business_description or "Unknown",
                    target_context=target_table.business_description or "Unknown",
                )
                
                result = self._call_llm(prompt)
                if result:
                    rel.business_description = result.get("business_description")
                    rel.data_flow_description = result.get("data_flow")
                    
            except Exception as e:
                logger.warning(f"Failed to analyze relationship {rel.source_table} -> {rel.target_table}: {e}")
    
    def _generate_use_cases(self, store: IntelligenceStore):
        """Generate business use cases from the schema."""
        logger.info("Generating business use cases...")
        
        try:
            # Build schema summary
            schema_summary = self._build_schema_summary(store)
            
            # Build table purposes
            table_purposes = "\n".join([
                f"- {name}: {table.business_description or 'Unknown purpose'}"
                for name, table in store.tables.items()
            ])
            
            # Build relationships summary
            relationships = "\n".join([
                f"- {r.source_table} -> {r.target_table} ({r.cardinality})"
                for r in store.relationships
            ])
            
            prompt = PromptTemplates.USE_CASE_INFERENCE_PROMPT.format(
                schema_summary=schema_summary,
                table_purposes=table_purposes,
                relationships=relationships,
            )
            
            result = self._call_llm(prompt)
            
            if result:
                # Add business questions as insights
                for q in result.get("business_questions", []):
                    store.add_insight(BusinessInsight(
                        category="BusinessQuestion",
                        title=q.get("question", "")[:50],
                        description=q.get("question", ""),
                        related_tables=q.get("relevant_tables", []),
                        priority=q.get("complexity", "medium"),
                    ))
                
                # Add KPIs
                for kpi in result.get("suggested_kpis", []):
                    store.add_insight(BusinessInsight(
                        category="KPI",
                        title=kpi.get("name", ""),
                        description=kpi.get("description", ""),
                        related_tables=kpi.get("relevant_tables", []),
                    ))
                
                # Add use cases
                for uc in result.get("analytics_use_cases", []):
                    store.add_insight(BusinessInsight(
                        category="UseCase",
                        title=f"{uc.get('domain', '')}: {uc.get('use_case', '')}",
                        description=uc.get("description", ""),
                        related_tables=uc.get("tables_involved", []),
                    ))
                    
        except Exception as e:
            logger.warning(f"Failed to generate use cases: {e}")
    
    def _generate_executive_summary(self, store: IntelligenceStore):
        """Generate executive summary."""
        logger.info("Generating executive summary...")
        
        try:
            stats = store.get_statistics()
            
            table_summary = "\n".join([
                f"- {name} ({table.table_type.value}): {table.row_count:,} rows, "
                f"{table.business_entity or 'entity unknown'}"
                for name, table in store.tables.items()
            ])
            
            relationship_summary = "\n".join([
                f"- {r.source_table} -> {r.target_table}: {r.business_description or r.cardinality}"
                for r in store.relationships[:10]  # Limit to first 10
            ])
            
            prompt = PromptTemplates.format_executive_summary(
                db_type=store.database_type or "Unknown",
                total_tables=stats["total_tables"],
                total_rows=stats["total_rows"],
                total_relationships=stats["total_relationships"],
                table_summary=table_summary,
                relationship_summary=relationship_summary,
                quality_score=stats["overall_quality_score"],
                critical_issues=stats["critical_issues"],
                warning_issues=stats["warning_issues"],
            )
            
            result = self._call_llm(prompt, expect_json=False)
            
            if result:
                store.executive_summary = result.get("text", "")
                
        except Exception as e:
            logger.warning(f"Failed to generate executive summary: {e}")
    
    def _generate_improvement_suggestions(self, store: IntelligenceStore):
        """Generate improvement suggestions."""
        logger.info("Generating improvement suggestions...")
        
        try:
            schema_summary = self._build_schema_summary(store)
            
            issues = "\n".join([
                f"- [{i.severity.value}] {i.table}{('.' + i.column) if i.column else ''}: {i.description}"
                for i in store.quality_issues[:20]  # Limit
            ])
            
            patterns = f"""
            - Core tables: {', '.join(store.get_core_tables()[:5])}
            - Transaction tables: {', '.join(store.get_transaction_tables()[:5])}
            - Isolated tables: {', '.join(store.get_isolated_tables()[:5])}
            """
            
            prompt = PromptTemplates.IMPROVEMENT_SUGGESTIONS_PROMPT.format(
                schema_summary=schema_summary,
                issues=issues or "No significant issues found",
                patterns=patterns,
            )
            
            result = self._call_llm(prompt)
            
            if result:
                store.improvement_suggestions = [
                    f"[{s.get('priority', 'medium')}] {s.get('suggestion', '')}"
                    for s in result.get("schema_improvements", [])
                ]
                
                store.warnings_and_caveats.extend(
                    result.get("governance_recommendations", [])
                )
                
        except Exception as e:
            logger.warning(f"Failed to generate improvement suggestions: {e}")
    
    def _build_schema_summary(self, store: IntelligenceStore) -> str:
        """Build a concise schema summary for prompts."""
        lines = []
        for name, table in store.tables.items():
            cols = ", ".join(list(table.columns.keys())[:5])
            if len(table.columns) > 5:
                cols += f", ... ({len(table.columns)} total)"
            lines.append(f"{name}: {cols}")
        return "\n".join(lines)
    
    def generate_sample_queries(
        self, 
        store: IntelligenceStore, 
        questions: List[str]
    ) -> List[Dict[str, str]]:
        """
        Generate sample SQL queries for given business questions.
        
        Args:
            store: Intelligence store with schema information
            questions: List of business questions
            
        Returns:
            List of query dictionaries with question, sql, explanation
        """
        if not self._client:
            return []
        
        try:
            schema_info = self._build_detailed_schema(store)
            
            prompt = PromptTemplates.SAMPLE_QUERY_PROMPT.format(
                db_type=store.database_type or "SQL",
                schema_info=schema_info,
                questions="\n".join([f"- {q}" for q in questions]),
            )
            
            result = self._call_llm(prompt)
            
            if result:
                return result.get("queries", [])
                
        except Exception as e:
            logger.warning(f"Failed to generate sample queries: {e}")
        
        return []
    
    def _build_detailed_schema(self, store: IntelligenceStore) -> str:
        """Build detailed schema information for query generation."""
        lines = []
        for name, table in store.tables.items():
            cols = []
            for col_name, col in table.columns.items():
                flags = []
                if col.is_primary_key:
                    flags.append("PK")
                if col.is_foreign_key:
                    flags.append(f"FK->{col.references_table}")
                flag_str = f" [{', '.join(flags)}]" if flags else ""
                cols.append(f"    {col_name} {col.data_type}{flag_str}")
            
            lines.append(f"TABLE {name}:")
            lines.extend(cols)
            lines.append("")
        
        return "\n".join(lines)
