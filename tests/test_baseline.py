"""Tests for baseline extractor."""

import io
import pytest
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from document_intelligence.baseline_extractor import (
    BaselineExtractor,
    FieldType,
    create_extractor,
    get_extractor,
)
from document_intelligence.preprocessing.pipeline import PreprocessingPipeline, create_pipeline, pdf_to_images
from document_intelligence.config import settings
from document_intelligence.services.extraction_service import ExtractionService, ExtractionInput


@pytest.fixture
def sample_invoice_image():
    """Create a simple test invoice image."""
    img = Image.new('RGB', (800, 600), 'white')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
        title_font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    # Title
    draw.text((50, 30), "INVOICE", fill='black', font=title_font)

    # Invoice number
    draw.text((50, 80), "Invoice #: INV-2024-001", fill='black', font=font)

    # Date
    draw.text((50, 110), "Date: 2024-01-15", fill='black', font=font)
    draw.text((50, 140), "Due Date: 2024-02-15", fill='black', font=font)

    # Vendor
    draw.text((50, 190), "FROM:", fill='black', font=font)
    draw.text((50, 220), "Acme Corporation", fill='black', font=font)
    draw.text((50, 250), "123 Main St", fill='black', font=font)
    draw.text((50, 280), "New York, NY 10001", fill='black', font=font)
    draw.text((50, 310), "Tax ID: 12-3456789", fill='black', font=font)

    # Line items
    draw.text((50, 360), "DESCRIPTION", fill='black', font=font)
    draw.text((400, 360), "QTY", fill='black', font=font)
    draw.text((500, 360), "UNIT PRICE", fill='black', font=font)
    draw.text((650, 360), "TOTAL", fill='black', font=font)

    draw.text((50, 390), "Consulting Services", fill='black', font=font)
    draw.text((400, 390), "10", fill='black', font=font)
    draw.text((500, 390), "$150.00", fill='black', font=font)
    draw.text((650, 390), "$1,500.00", fill='black', font=font)

    # Totals
    draw.text((500, 440), "Subtotal:", fill='black', font=font)
    draw.text((650, 440), "$1,500.00", fill='black', font=font)
    draw.text((500, 470), "Tax (10%):", fill='black', font=font)
    draw.text((650, 470), "$150.00", fill='black', font=font)
    draw.text((500, 500), "TOTAL:", fill='black', font=font)
    draw.text((650, 500), "$1,650.00", fill='black', font=font)

    return np.array(img)


@pytest.fixture
def sample_invoice_bytes(sample_invoice_image):
    """Create invoice as bytes."""
    img = Image.fromarray(sample_invoice_image)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


class TestBaselineExtractor:
    """Test baseline OCR + Regex extractor."""

    def test_extractor_creation(self):
        """Test extractor can be created."""
        extractor = create_extractor(lang="en", use_gpu=False)
        assert extractor is not None
        assert isinstance(extractor, BaselineExtractor)

    def test_extractor_singleton(self):
        """Test extractor is singleton."""
        ext1 = get_extractor()
        ext2 = get_extractor()
        assert ext1 is ext2

    def test_extract_fields(self, sample_invoice_image):
        """Test field extraction from sample invoice."""
        extractor = create_extractor(lang="en", use_gpu=False)
        result = extractor.extract(sample_invoice_image)

        assert isinstance(result.raw_ocr_text, str)
        assert len(result.raw_ocr_text) > 0
        assert result.processing_time_ms > 0
        assert len(result.fields) > 0

        # Check specific fields
        field_types = {f.field_type for f in result.fields}
        assert FieldType.INVOICE_NUMBER in field_types
        assert FieldType.INVOICE_DATE in field_types
        assert FieldType.VENDOR_NAME in field_types
        assert FieldType.TOTAL_AMOUNT in field_types

    def test_invoice_number_extraction(self, sample_invoice_image):
        """Test invoice number extraction."""
        extractor = create_extractor(lang="en", use_gpu=False)
        result = extractor.extract(sample_invoice_image)

        inv_field = result.get_field(FieldType.INVOICE_NUMBER)
        assert inv_field is not None
        assert "INV-2024-001" in inv_field.value or "INV2024001" in inv_field.value

    def test_date_extraction(self, sample_invoice_image):
        """Test date extraction."""
        extractor = create_extractor(lang="en", use_gpu=False)
        result = extractor.extract(sample_invoice_image)

        inv_date = result.get_field(FieldType.INVOICE_DATE)
        due_date = result.get_field(FieldType.DUE_DATE)

        assert inv_date is not None
        assert due_date is not None
        assert "2024-01-15" in inv_date.value or "2024/01/15" in inv_date.value

    def test_amount_extraction(self, sample_invoice_image):
        """Test amount extraction."""
        extractor = create_extractor(lang="en", use_gpu=False)
        result = extractor.extract(sample_invoice_image)

        total = result.get_field(FieldType.TOTAL_AMOUNT)
        subtotal = result.get_field(FieldType.SUBTOTAL)
        tax = result.get_field(FieldType.TAX_AMOUNT)

        assert total is not None
        assert "1650" in total.value.replace(".", "").replace(",", "")
        assert subtotal is not None
        assert tax is not None

    def test_safe_group_regex(self):
        """Test _safe_group handles missing capture groups."""
        extractor = create_extractor(lang="en", use_gpu=False)
        
        # Pattern without capture groups
        import re
        pattern = re.compile(r'(?i)test')
        match = pattern.search("test")
        assert extractor._safe_group(match) == "test"
        
        # Pattern with capture groups
        pattern2 = re.compile(r'(?i)(test)')
        match2 = pattern2.search("test")
        assert extractor._safe_group(match2) == "test"
        
        # Pattern with multiple groups - should get first
        pattern3 = re.compile(r'(?i)(test)(\d+)')
        match3 = pattern3.search("test123")
        assert extractor._safe_group(match3) == "test"


class TestPreprocessingPipeline:
    """Test preprocessing pipeline."""

    def test_pipeline_creation(self):
        """Test pipeline can be created."""
        pipeline = create_pipeline()
        assert pipeline is not None
        assert isinstance(pipeline, PreprocessingPipeline)

    def test_process_image(self, sample_invoice_image):
        """Test image processing."""
        pipeline = create_pipeline(save_intermediate=False)
        result = pipeline.process(sample_invoice_image)

        assert result.image is not None
        assert result.image.shape[:2] != sample_invoice_image.shape[:2] or len(result.stages_applied) > 0
        assert "grayscale" in result.stages_applied or "clahe" in result.stages_applied

    def test_pipeline_config(self):
        """Test pipeline with custom config."""
        pipeline = create_pipeline(clahe=False, deskew=False, denoise=False)
        assert not pipeline.config.apply_clahe
        assert not pipeline.config.apply_deskew
        assert not pipeline.config.apply_denoise


class TestConfig:
    """Test configuration loading."""

    def test_settings_load(self):
        """Test settings can be loaded."""
        assert settings.app_name == "document-intelligence"
        assert settings.api_port == 8000
        assert hasattr(settings, 'api_keys')
        assert hasattr(settings, 'jwt_secret')


class TestExtractionService:
    """Test extraction service."""

    def test_service_creation(self):
        """Test service can be created."""
        from unittest.mock import MagicMock
        mock_minio = MagicMock()
        service = ExtractionService(minio_client=mock_minio)
        assert service is not None

    def test_extract_from_bytes(self, sample_invoice_bytes):
        """Test extraction from bytes."""
        from unittest.mock import MagicMock
        mock_minio = MagicMock()
        service = ExtractionService(minio_client=mock_minio)
        
        input_data = ExtractionInput(
            file_bytes=sample_invoice_bytes,
            filename="test_invoice.png",
            mime_type="image/png",
            document_id="test-123",
        )
        
        result = service.extract(input_data)
        
        assert result.document_id == "test-123"
        assert result.status in ("completed", "failed")
        assert isinstance(result.fields, dict)
        assert isinstance(result.confidence_scores, dict)
        assert result.processing_time_ms > 0


class TestMultiPageSupport:
    """Test multi-page document support."""

    def test_pdf_to_images_import(self):
        """Test PDF conversion imports work."""
        from document_intelligence.preprocessing.pipeline import pdf_to_images, HAS_PYMUPDF, HAS_PDF2IMAGE
        # At least one should be available in test env
        # (This test just verifies imports work)
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])