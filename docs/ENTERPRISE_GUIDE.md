# 🏢 Enterprise Deployment Guide

## Extending DataMind AI to Enterprise Databases

This guide explains how to scale DataMind AI from a demo environment to production enterprise databases with 100+ tables.

---

## Architecture for Scale

### Horizontal Scaling Strategy

```
                                    ┌──────────────────┐
                                    │   Load Balancer  │
                                    └────────┬─────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
           ┌────────┴────────┐      ┌────────┴────────┐      ┌────────┴────────┐
           │  Worker Node 1  │      │  Worker Node 2  │      │  Worker Node N  │
           │  ┌───────────┐  │      │  ┌───────────┐  │      │  ┌───────────┐  │
           │  │  Scanner  │  │      │  │  Scanner  │  │      │  │  Scanner  │  │
           │  │  Profiler │  │      │  │  Profiler │  │      │  │  Profiler │  │
           │  │  Analyzer │  │      │  │  Analyzer │  │      │  │  Analyzer │  │
           │  └───────────┘  │      │  └───────────┘  │      │  └───────────┘  │
           └────────┬────────┘      └────────┬────────┘      └────────┬────────┘
                    │                        │                        │
                    └────────────────────────┼────────────────────────┘
                                             │
                                    ┌────────┴─────────┐
                                    │   Redis Queue    │
                                    │   (Task Queue)   │
                                    └────────┬─────────┘
                                             │
                              ┌──────────────┴──────────────┐
                              │                             │
                     ┌────────┴────────┐           ┌────────┴────────┐
                     │  LLM Inference  │           │  Documentation  │
                     │     Engine      │           │    Generator    │
                     └────────┬────────┘           └────────┬────────┘
                              │                             │
                              └──────────────┬──────────────┘
                                             │
                                    ┌────────┴─────────┐
                                    │  Intelligence    │
                                    │     Store        │
                                    │   (PostgreSQL)   │
                                    └──────────────────┘
```

---

## Deployment Options

### Option 1: Docker Compose (Small/Medium)

```yaml
# docker-compose.yml
version: '3.8'

services:
  datamind-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@postgres:5432/datamind
    depends_on:
      - redis
      - postgres

  datamind-worker:
    build: .
    command: python -m celery -A datamind.tasks worker
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=datamind
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  redis_data:
  postgres_data:
```

### Option 2: Kubernetes (Large Scale)

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: datamind-scanner
spec:
  replicas: 5  # Scale based on database count
  selector:
    matchLabels:
      app: datamind-scanner
  template:
    spec:
      containers:
      - name: scanner
        image: datamind-ai:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: WORKER_TYPE
          value: "scanner"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: datamind-llm
spec:
  replicas: 3  # LLM workers
  selector:
    matchLabels:
      app: datamind-llm
  template:
    spec:
      containers:
      - name: llm-worker
        image: datamind-ai:latest
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
        env:
        - name: WORKER_TYPE
          value: "llm"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: datamind-secrets
              key: openai-api-key
```

---

## Handling Large Databases

### Chunked Processing

For databases with 100+ tables:

```python
# datamind/core/chunked_processor.py

class ChunkedProcessor:
    """Process large databases in manageable chunks."""
    
    def __init__(self, config: DataMindConfig, chunk_size: int = 10):
        self.config = config
        self.chunk_size = chunk_size
    
    async def process_database(self, tables: List[str]) -> IntelligenceStore:
        """Process tables in parallel chunks."""
        store = IntelligenceStore()
        
        # Split tables into chunks
        chunks = [tables[i:i + self.chunk_size] 
                  for i in range(0, len(tables), self.chunk_size)]
        
        # Process chunks in parallel
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(self._process_chunk(chunk, store))
                for chunk in chunks
            ]
        
        return store
    
    async def _process_chunk(
        self, 
        tables: List[str], 
        store: IntelligenceStore
    ):
        """Process a chunk of tables."""
        for table in tables:
            profile = await self._scan_and_profile(table)
            store.add_table(profile)
```

### Incremental Updates

For frequently changing databases:

```python
# datamind/core/incremental_analyzer.py

class IncrementalAnalyzer:
    """Support incremental database analysis."""
    
    def __init__(self, cache_dir: str = ".datamind_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_changed_tables(
        self, 
        conn_manager: DatabaseConnectionManager,
        last_analysis: datetime
    ) -> List[str]:
        """Identify tables that have changed since last analysis."""
        # For PostgreSQL - use pg_stat_user_tables
        query = """
            SELECT schemaname, relname, n_tup_ins, n_tup_upd, n_tup_del, 
                   last_vacuum, last_analyze
            FROM pg_stat_user_tables
            WHERE last_analyze > %s OR last_vacuum > %s
        """
        return conn_manager.execute_query(query, (last_analysis, last_analysis))
    
    def load_cached_analysis(self, table_name: str) -> Optional[TableProfile]:
        """Load previous analysis from cache."""
        cache_file = self.cache_dir / f"{table_name}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return TableProfile.from_dict(json.load(f))
        return None
    
    def save_to_cache(self, table: TableProfile):
        """Cache table analysis for future incremental runs."""
        cache_file = self.cache_dir / f"{table.name}.json"
        with open(cache_file, 'w') as f:
            json.dump(table.to_dict(), f)
```

---

## LLM Cost Optimization

### Batching Strategy

```python
# datamind/inference/batch_processor.py

class LLMBatchProcessor:
    """Optimize LLM calls through intelligent batching."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.batch_size = 5  # Columns per call
        self.max_concurrent = 3  # Parallel LLM calls
    
    async def process_columns_batch(
        self, 
        columns: List[ColumnProfile]
    ) -> Dict[str, Dict]:
        """Process multiple columns in optimized batches."""
        results = {}
        
        # Group similar columns
        batches = self._create_smart_batches(columns)
        
        # Process with rate limiting
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_batch(batch):
            async with semaphore:
                return await self._call_llm_batch(batch)
        
        tasks = [process_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks)
        
        for result in batch_results:
            results.update(result)
        
        return results
    
    def _create_smart_batches(
        self, 
        columns: List[ColumnProfile]
    ) -> List[List[ColumnProfile]]:
        """Create batches grouping similar columns."""
        # Group by semantic type for better context
        by_type = defaultdict(list)
        for col in columns:
            by_type[col.semantic_type].append(col)
        
        batches = []
        for cols in by_type.values():
            for i in range(0, len(cols), self.batch_size):
                batches.append(cols[i:i + self.batch_size])
        
        return batches
```

### Token Budget Management

```python
# datamind/inference/token_manager.py

class TokenBudgetManager:
    """Manage LLM token usage within budget constraints."""
    
    def __init__(self, daily_budget: int = 1_000_000):
        self.daily_budget = daily_budget
        self.tokens_used = 0
        self.reset_time = datetime.now()
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Rough estimate: 4 chars per token
        return len(text) // 4
    
    def can_process(self, estimated_tokens: int) -> bool:
        """Check if we have budget for this request."""
        self._maybe_reset()
        return (self.tokens_used + estimated_tokens) <= self.daily_budget
    
    def record_usage(self, tokens: int):
        """Record token usage."""
        self.tokens_used += tokens
    
    def _maybe_reset(self):
        """Reset budget if new day."""
        if (datetime.now() - self.reset_time).days >= 1:
            self.tokens_used = 0
            self.reset_time = datetime.now()
```

---

## Multi-Database Support

### Database Registry

```python
# datamind/core/database_registry.py

class DatabaseRegistry:
    """Manage multiple database connections."""
    
    def __init__(self, config_path: str):
        self.databases: Dict[str, DatabaseConfig] = {}
        self._load_config(config_path)
    
    def _load_config(self, path: str):
        """Load database configurations from YAML."""
        with open(path) as f:
            config = yaml.safe_load(f)
        
        for db_name, db_config in config.get('databases', {}).items():
            self.databases[db_name] = DatabaseConfig(
                db_type=DatabaseType(db_config['type']),
                host=db_config.get('host'),
                port=db_config.get('port'),
                database=db_config.get('database'),
                username=db_config.get('username'),
                password=db_config.get('password'),
            )
    
    def get_connection(self, db_name: str) -> DatabaseConnectionManager:
        """Get connection manager for named database."""
        if db_name not in self.databases:
            raise ValueError(f"Unknown database: {db_name}")
        return DatabaseConnectionManager(self.databases[db_name])
    
    async def analyze_all(self) -> Dict[str, IntelligenceStore]:
        """Analyze all registered databases."""
        results = {}
        
        async with asyncio.TaskGroup() as tg:
            for db_name in self.databases:
                task = tg.create_task(self._analyze_database(db_name))
                results[db_name] = task
        
        return {name: task.result() for name, task in results.items()}
```

### Configuration Example

```yaml
# databases.yaml
databases:
  production_main:
    type: postgresql
    host: prod-db.company.com
    port: 5432
    database: main_app
    username: ${PROD_DB_USER}
    password: ${PROD_DB_PASS}
    
  analytics_warehouse:
    type: postgresql
    host: analytics-db.company.com
    port: 5432
    database: warehouse
    username: ${ANALYTICS_USER}
    password: ${ANALYTICS_PASS}
    
  legacy_mysql:
    type: mysql
    host: legacy-db.company.com
    port: 3306
    database: legacy_app
    username: ${LEGACY_USER}
    password: ${LEGACY_PASS}

settings:
  parallel_databases: 3
  cache_enabled: true
  incremental: true
```

---

## API Integration

### REST API Endpoints

```python
# datamind/api/routes.py

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

app = FastAPI(title="DataMind AI API")

class AnalysisRequest(BaseModel):
    database_name: str
    skip_llm: bool = False
    tables: Optional[List[str]] = None

class AnalysisResponse(BaseModel):
    job_id: str
    status: str
    message: str

@app.post("/analyze", response_model=AnalysisResponse)
async def start_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
):
    """Start database analysis job."""
    job_id = str(uuid.uuid4())
    
    background_tasks.add_task(
        run_analysis_job,
        job_id=job_id,
        database_name=request.database_name,
        skip_llm=request.skip_llm,
        tables=request.tables
    )
    
    return AnalysisResponse(
        job_id=job_id,
        status="started",
        message="Analysis job queued"
    )

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get analysis job status."""
    status = await get_job_status(job_id)
    return status

@app.get("/documentation/{job_id}")
async def get_documentation(job_id: str):
    """Download generated documentation."""
    doc_path = await get_job_output(job_id)
    return FileResponse(doc_path)
```

---

## Monitoring & Observability

### Prometheus Metrics

```python
# datamind/monitoring/metrics.py

from prometheus_client import Counter, Histogram, Gauge

# Counters
tables_analyzed = Counter(
    'datamind_tables_analyzed_total',
    'Total tables analyzed',
    ['database', 'status']
)

llm_calls = Counter(
    'datamind_llm_calls_total',
    'Total LLM API calls',
    ['model', 'status']
)

# Histograms
analysis_duration = Histogram(
    'datamind_analysis_duration_seconds',
    'Time spent analyzing databases',
    ['database'],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600]
)

llm_latency = Histogram(
    'datamind_llm_latency_seconds',
    'LLM API call latency',
    ['model'],
    buckets=[0.5, 1, 2, 5, 10, 30]
)

# Gauges
active_analyses = Gauge(
    'datamind_active_analyses',
    'Number of analyses currently running'
)

quality_score = Gauge(
    'datamind_quality_score',
    'Latest data quality score',
    ['database']
)
```

---

## Security Considerations

### Credential Management

```python
# datamind/security/credentials.py

import hvac  # HashiCorp Vault client

class SecureCredentialManager:
    """Manage database credentials securely."""
    
    def __init__(self, vault_url: str, vault_token: str):
        self.client = hvac.Client(url=vault_url, token=vault_token)
    
    def get_database_credentials(
        self, 
        database_name: str
    ) -> DatabaseConfig:
        """Retrieve credentials from Vault."""
        secret = self.client.secrets.kv.v2.read_secret_version(
            path=f"datamind/databases/{database_name}"
        )
        
        data = secret['data']['data']
        
        return DatabaseConfig(
            db_type=DatabaseType(data['type']),
            host=data['host'],
            port=data['port'],
            database=data['database'],
            username=data['username'],
            password=data['password'],
        )
```

### Data Masking

```python
# datamind/security/masking.py

class DataMasker:
    """Mask sensitive data in sample values."""
    
    PII_PATTERNS = {
        'email': r'[\w\.-]+@[\w\.-]+\.\w+',
        'phone': r'\+?\d{10,}',
        'ssn': r'\d{3}-\d{2}-\d{4}',
        'credit_card': r'\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}',
    }
    
    def mask_samples(
        self, 
        samples: List[Any],
        column_name: str
    ) -> List[str]:
        """Mask potentially sensitive sample values."""
        masked = []
        for sample in samples:
            sample_str = str(sample)
            
            # Check for PII patterns
            for pii_type, pattern in self.PII_PATTERNS.items():
                if re.search(pattern, sample_str):
                    sample_str = f"[MASKED {pii_type.upper()}]"
                    break
            
            masked.append(sample_str)
        
        return masked
```

---

## Performance Benchmarks

### Expected Performance

| Database Size | Tables | Scan Time | Profile Time | LLM Time | Total |
|--------------|--------|-----------|--------------|----------|-------|
| Small | 10 | 2s | 5s | 30s | ~40s |
| Medium | 50 | 10s | 30s | 3m | ~4m |
| Large | 100 | 25s | 90s | 8m | ~10m |
| Enterprise | 500 | 2m | 8m | 30m | ~40m |

*LLM time can be reduced with parallel processing and caching*

### Optimization Techniques

1. **Connection Pooling**: Reuse database connections
2. **Parallel Scanning**: Scan multiple tables concurrently
3. **Smart Sampling**: Profile only necessary data
4. **LLM Caching**: Cache semantic descriptions
5. **Incremental Updates**: Only re-analyze changed tables

---

## Support & Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Slow scanning | Large tables | Enable sampling |
| LLM timeout | Complex schemas | Increase batch size |
| Memory issues | Too many columns | Use chunked processing |
| Connection errors | Pool exhaustion | Increase pool size |

### Logging Configuration

```yaml
# logging.yaml
version: 1
disable_existing_loggers: false

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    
  file:
    class: logging.handlers.RotatingFileHandler
    filename: datamind.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
    level: DEBUG

loggers:
  datamind:
    level: DEBUG
    handlers: [console, file]
    propagate: false
```

---

## Conclusion

DataMind AI is designed to scale from small demo databases to enterprise-grade deployments. By following this guide, you can:

1. Deploy on Kubernetes for high availability
2. Process 100+ table databases efficiently
3. Manage LLM costs with intelligent batching
4. Support multiple databases simultaneously
5. Integrate with existing monitoring infrastructure

For support, contact: enterprise@datamind.ai
