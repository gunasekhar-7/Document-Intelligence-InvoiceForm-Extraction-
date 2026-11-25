# Document Intelligence: Invoice & Form Extraction

> A production-ready document intelligence system for automated invoice and form extraction using LayoutLMv3, PaddleOCR, and FastAPI.

## 🎯 Overview

This project demonstrates end-to-end ML system engineering for document understanding:
- **Computer Vision**: OpenCV preprocessing (deskew, denoise, binarization, line removal)
- **OCR**: PaddleOCR for text detection & recognition
- **NLP/Layout Understanding**: LayoutLMv3 fine-tuned for token classification (NER)
- **Backend**: Async FastAPI with PostgreSQL, Redis, MinIO
- **MLOps**: MLflow tracking, DVC data versioning, ONNX export
- **Production**: Docker, Kubernetes (Helm), Prometheus/Grafana monitoring, CI/CD

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Client    │────▶│  FastAPI     │────▶│  PostgreSQL │
│  (Upload)   │     │  /ingest     │     │  (Metadata) │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
                    ┌──────▼───────┐     ┌─────────────┐
                    │  Redis Queue │────▶│  Worker     │
                    │  (ARQ)       │     │  (Extract)  │
                    └──────────────┘     └──────┬──────┘
                                                │
                                         ┌──────▼──────┐
                                         │  MinIO/S3   │
                                         │  (Artifacts)│
                                         └─────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU (for worker) or CPU-only mode
- 16GB+ RAM recommended

### Development Setup

```bash
# Clone and navigate
git clone <repo-url>
cd document-intelligence

# Copy environment template
cp .env.example .env

# Start all services (CPU mode)
docker-compose up -d

# Or with GPU support (requires nvidia-docker)
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Access services
# API: http://localhost:8000/docs
# MinIO Console: http://localhost:9001 (minioadmin/minioadmin)
# MLflow: http://localhost:5000
# Demo UI: http://localhost:8501 (run: streamlit run demo/app.py)
```

### Generate Synthetic Data

```bash
# Generate 1000 synthetic invoices
python -m document_intelligence.data.synthetic.generate \
  --output data/synthetic \
  --count 1000
```

### Train LayoutLMv3

```bash
# Fine-tune on synthetic + public data
python -m document_intelligence.training.train \
  --train-data data/synthetic/train_annotations.json \
  --val-data data/synthetic/val_annotations.json \
  --output-dir models/layoutlmv3-finetuned \
  --epochs 10 \
  --batch-size 4 \
  --lr 3e-5
```

### Run Evaluation

```bash
# Compare baseline vs fine-tuned model
python -m document_intelligence.evaluation.metrics \
  --predictions results/baseline_predictions.json \
  --ground-truth data/test/ground_truth.json \
  --output evaluation/baseline_report.html \
  --model-name "baseline"

python -m document_intelligence.evaluation.metrics \
  --predictions results/model_predictions.json \
  --ground-truth data/test/ground_truth.json \
  --output evaluation/model_report.html \
  --model-name "layoutlmv3" \
  --baseline evaluation/baseline_results.json
```

## 📁 Project Structure

```
document-intelligence/
├── src/document_intelligence/
│   ├── api/              # FastAPI application
│   │   ├── main.py       # App factory
│   │   └── routes/       # API endpoints
│   ├── workers/          # ARQ async workers
│   ├── services/         # Business logic (extraction, storage)
│   ├── db/               # SQLAlchemy models & sessions
│   ├── preprocessing/    # OpenCV pipeline
│   ├── training/         # LayoutLMv3 fine-tuning
│   ├── inference/        # Model serving (ONNX/TensorRT)
│   ├── evaluation/       # Metrics & error analysis
│   ├── baseline_extractor.py  # OCR + Regex baseline
│   ├── config.py         # Settings management
│   └── logging.py        # Structured logging
├── data/
│   ├── synthetic/        # Generated invoices
│   ├── public/           # SROIE, CORD, FUNSD
│   └── processed/        # Preprocessed datasets
├── docker/               # Dockerfiles
├── helm/                 # Kubernetes Helm chart
├── monitoring/           # Prometheus/Grafana configs
├── demo/                 # Streamlit demo app
├── tests/                # Unit & integration tests
├── docker-compose.yml    # Local development stack
├── requirements.txt      # Python dependencies
└── pyproject.toml        # Project configuration
```

## 🔧 Configuration

Key environment variables (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection | `postgresql+asyncpg://...` |
| `REDIS_URL` | Redis connection | `redis://localhost:6379/0` |
| `MINIO_ENDPOINT` | MinIO endpoint | `localhost:9000` |
| `MODEL_PATH` | Model directory | `models/best_model` |
| `OCR_ENGINE` | OCR backend | `paddleocr` |
| `LOG_LEVEL` | Logging level | `INFO` |

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/documents` | Upload document for extraction |
| `GET` | `/api/v1/documents` | List documents (paginated, filtered) |
| `GET` | `/api/v1/documents/{id}` | Get document metadata |
| `GET` | `/api/v1/documents/{id}/extraction` | Get extraction results |
| `GET` | `/api/v1/documents/{id}/download` | Download original file |
| `DELETE` | `/api/v1/documents/{id}` | Delete document |
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/api/v1/health/ready` | Readiness probe |
| `GET` | `/api/v1/health/live` | Liveness probe |

### Example Request

```bash
curl -X POST "http://localhost:8000/api/v1/documents" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@invoice.pdf"
```

### Example Response

```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "message": "Document processed successfully"
}
```

## 🧪 Testing

```bash
# Unit tests
pytest tests/unit -v

# Integration tests (requires running services)
pytest tests/integration -v

# Load testing
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

## 📈 Monitoring

- **Prometheus**: Metrics collection (`:9090`)
- **Grafana**: Dashboards (`:3000`, admin/admin)
- **Loki**: Log aggregation (`:3100`)
- **Tempo**: Distributed tracing (`:3200`)

Key dashboards:
- API Latency (p50, p95, p99)
- Error Rates
- Queue Depth
- Processing Time
- GPU Memory Utilization
- Extraction F1 Score

## 🚢 Deployment

### Kubernetes (Helm)

```bash
# Install
helm install document-intelligence ./helm/document-intelligence \
  --namespace document-intelligence \
  --create-namespace \
  --set global.environment=production \
  --set secrets.databaseUrl=... \
  --set secrets.redisUrl=... \
  --set secrets.minioAccessKey=... \
  --set secrets.minioSecretKey=...

# Upgrade
helm upgrade document-intelligence ./helm/document-intelligence \
  --namespace document-intelligence \
  --set image.tag=v1.2.3
```

### Production Checklist

- [ ] Use managed PostgreSQL (RDS, Cloud SQL)
- [ ] Use managed Redis (ElastiCache, Memorystore)
- [ ] Use managed object storage (S3, GCS)
- [ ] Configure TLS certificates (cert-manager)
- [ ] Set up external secrets operator
- [ ] Configure log aggregation (Loki/ELK)
- [ ] Set up alerting (Alertmanager → PagerDuty/Slack)
- [ ] Enable network policies
- [ ] Configure pod disruption budgets
- [ ] Set resource limits/requests appropriately

## 📚 Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **LayoutLMv3 over Donut** | Multimodal (text+layout+image), better for structured docs |
| **PaddleOCR over Tesseract** | Superior table/layout detection, active development |
| **FastAPI over Flask/Django** | Async-native, automatic OpenAPI, type safety |
| **ARQ over Celery** | Native async, Redis-native, simpler |
| **MinIO over S3 API** | S3-compatible, self-hosted, no vendor lock-in |
| **MLflow over Weights&Biases** | Open-source, self-hosted, integrated with DVC |
| **ONNX Runtime for serving** | Cross-platform, TensorRT optimization, no Python in prod |

## 🎓 Learning Outcomes (Resume Highlights)

- **ML System Design**: End-to-end pipeline from data → model → API → production
- **Computer Vision**: OpenCV preprocessing pipeline handling real-world noise
- **NLP/Layout Understanding**: LayoutLMv3 fine-tuning with custom token-bbox alignment
- **Backend Engineering**: Async FastAPI with PostgreSQL, Redis, MinIO
- **MLOps**: MLflow experiment tracking, DVC data versioning, ONNX export
- **DevOps**: Docker multi-stage builds, Helm charts, GitHub Actions CI/CD
- **Observability**: Prometheus/Grafana/Loki/Tempo stack with custom alerts
- **Testing**: Unit, integration, load, and contract testing strategies

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a PR

## 📞 Support

- Issues: [GitHub Issues](https://github.com/yourusername/document-intelligence/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/document-intelligence/discussions)"# Document-Intelligence-InvoiceForm-Extraction-" 
