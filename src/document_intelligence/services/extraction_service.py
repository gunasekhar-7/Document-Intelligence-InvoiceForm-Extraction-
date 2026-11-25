"""Core extraction service - business logic for document processing."""

import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from document_intelligence.baseline_extractor import (
    BaselineExtractor,
    ExtractionResult,
    FieldType,
    create_extractor,
)
from document_intelligence.inference.model_server import (
    DocumentIntelligenceModelServer,
    get_model_server,
)
from document_intelligence.logging import get_logger
from document_intelligence.preprocessing.pipeline import (
    PreprocessingPipeline,
    create_pipeline,
    process_document_pages,
)
from document_intelligence.services.minio_client import (
    MinioClient,
    compute_content_hash,
    generate_object_key,
)

logger = get_logger(__name__)


@dataclass
class ExtractionInput:
    """Input for extraction pipeline."""

    file_bytes: bytes
    filename: str
    mime_type: str
    document_id: str | None = None
    priority: str = "normal"
    minio_key: str | None = None


@dataclass
class ExtractionOutput:
    """Output from extraction pipeline."""

    document_id: str
    fields: dict[str, Any]
    confidence_scores: dict[str, float]
    raw_ocr_text: str
    raw_entities: list[dict]
    processing_time_ms: int
    model_version: str
    ocr_engine: str
    minio_key: str
    status: str = "completed"
    error_message: str | None = None


class ExtractionService:
    """Production service for document extraction pipeline with neural + baseline hybrid engine."""

    def __init__(
        self,
        minio_client: MinioClient | None = None,
        extractor: BaselineExtractor | None = None,
        model_server: DocumentIntelligenceModelServer | None = None,
        preprocessor: PreprocessingPipeline | None = None,
        model_version: str = "hybrid-layoutlmv3-baseline",
    ):
        self.minio_client = minio_client
        self._extractor = extractor
        self._model_server = model_server
        self.preprocessor = preprocessor or create_pipeline()
        self.model_version = model_version

    @property
    def extractor(self) -> BaselineExtractor:
        if self._extractor is None:
            self._extractor = create_extractor()
        return self._extractor

    @property
    def model_server(self) -> DocumentIntelligenceModelServer:
        if self._model_server is None:
            self._model_server = get_model_server()
        return self._model_server

    def extract(self, input_data: ExtractionInput) -> ExtractionOutput:
        """Run full multi-page extraction pipeline with neural model + regex baseline."""
        start_time = time.perf_counter()
        doc_id = input_data.document_id or str(uuid.uuid4())

        try:
            content_hash = compute_content_hash(input_data.file_bytes)

            # Ensure object storage key
            minio_key = input_data.minio_key
            if not minio_key:
                minio_key = generate_object_key(input_data.filename, content_hash)
                if self.minio_client:
                    try:
                        self.minio_client.upload_bytes(minio_key, input_data.file_bytes)
                    except Exception as upload_err:
                        logger.warning("MinIO upload skipped or deferred", error=str(upload_err))

            # 1. Multi-page document preprocessing (supports PDF, PNG, JPG, TIFF)
            page_results = process_document_pages(
                input_data.file_bytes,
                input_data.mime_type,
                pipeline=self.preprocessor,
            )

            all_raw_ocr_texts = []
            combined_fields: dict[str, Any] = {}
            combined_confidence: dict[str, float] = {}
            all_entities: list[dict] = []

            # 2. Extract fields across pages (focusing on primary page for invoice metadata)
            for page_idx, page_res in enumerate(page_results):
                processed_image = page_res.image

                # Run baseline OCR
                extraction_result = self.extractor.extract(processed_image)
                if extraction_result.raw_ocr_text:
                    all_raw_ocr_texts.append(extraction_result.raw_ocr_text)

                page_fields, page_confidence = self._format_fields(extraction_result)

                # 3. If neural model is available, perform LayoutLMv3 prediction
                if self.model_server.is_loaded and extraction_result.raw_ocr_data:
                    words = []
                    boxes = []
                    for page in extraction_result.raw_ocr_data:
                        if page:
                            for line in page:
                                bbox, (text, _) = line
                                if text.strip():
                                    words.append(text)
                                    # bbox is [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
                                    xs = [p[0] for p in bbox]
                                    ys = [p[1] for p in bbox]
                                    boxes.append([min(xs), min(ys), max(xs), max(ys)])

                    if words:
                        neural_out = self.model_server.predict(processed_image, words, boxes)
                        neural_fields = neural_out.get("fields", {})
                        neural_conf = neural_out.get("confidence", {})

                        # High-confidence neural fields override or complement baseline
                        for k, v in neural_fields.items():
                            if v and neural_conf.get(k, 0) > page_confidence.get(k, 0):
                                page_fields[k] = v
                                page_confidence[k] = neural_conf[k]

                # Merge page fields (first page values take precedence unless missing)
                for k, v in page_fields.items():
                    if v and (k not in combined_fields or not combined_fields[k]):
                        combined_fields[k] = v
                        combined_confidence[k] = page_confidence.get(k, 0.7)

                all_entities.extend(self._format_entities(extraction_result))

            processing_time = int((time.perf_counter() - start_time) * 1000)

            return ExtractionOutput(
                document_id=doc_id,
                fields=combined_fields,
                confidence_scores=combined_confidence,
                raw_ocr_text="\n--- PAGE BREAK ---\n".join(all_raw_ocr_texts),
                raw_entities=all_entities,
                processing_time_ms=processing_time,
                model_version=self.model_version,
                ocr_engine="paddleocr",
                minio_key=minio_key or "",
                status="completed",
            )

        except Exception as e:
            logger.error("Extraction failed", document_id=doc_id, error=str(e))
            processing_time = int((time.perf_counter() - start_time) * 1000)
            return ExtractionOutput(
                document_id=doc_id,
                fields={},
                confidence_scores={},
                raw_ocr_text="",
                raw_entities=[],
                processing_time_ms=processing_time,
                model_version=self.model_version,
                ocr_engine="paddleocr",
                minio_key="",
                status="failed",
                error_message=str(e),
            )

    def _format_fields(self, result: ExtractionResult) -> tuple[dict[str, Any], dict[str, float]]:
        """Convert extraction result to structured field dictionary."""
        fields: dict[str, Any] = {}
        confidence: dict[str, float] = {}

        field_mapping = {
            FieldType.INVOICE_NUMBER: "invoice_number",
            FieldType.INVOICE_DATE: "invoice_date",
            FieldType.DUE_DATE: "due_date",
            FieldType.VENDOR_NAME: "vendor_name",
            FieldType.VENDOR_ADDRESS: "vendor_address",
            FieldType.VENDOR_TAX_ID: "vendor_tax_id",
            FieldType.SUBTOTAL: "subtotal",
            FieldType.TAX_AMOUNT: "tax_amount",
            FieldType.TOTAL_AMOUNT: "total_amount",
            FieldType.CURRENCY: "currency",
        }

        for field_type, key in field_mapping.items():
            extracted = result.get_field(field_type)
            if extracted:
                fields[key] = extracted.value
                confidence[key] = extracted.confidence

        # Normalize numeric amounts
        for key in ["subtotal", "tax_amount", "total_amount"]:
            if key in fields and fields[key] is not None:
                val = (
                    str(fields[key])
                    .replace(",", "")
                    .replace("$", "")
                    .replace("€", "")
                    .replace("£", "")
                    .strip()
                )
                try:
                    fields[key] = float(val)
                except ValueError:
                    pass

        # Normalize date format if feasible
        for key in ["invoice_date", "due_date"]:
            if key in fields and isinstance(fields[key], str):
                for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"):
                    try:
                        fields[key] = datetime.strptime(fields[key].strip(), fmt).date()
                        break
                    except ValueError:
                        continue

        return fields, confidence

    def _format_entities(self, result: ExtractionResult) -> list[dict]:
        """Format entities for database storage."""
        entities = []
        for field in result.fields:
            entities.append(
                {
                    "field_type": field.field_type.value,
                    "value": str(field.value),
                    "confidence": field.confidence,
                    "bbox": field.bbox,
                    "raw_text": field.raw_text,
                }
            )
        return entities
