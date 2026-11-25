"""Baseline OCR + Regex extraction for invoice fields."""

import re
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

try:
    from paddleocr import PaddleOCR

    HAS_PADDLEOCR = True
except ImportError:
    HAS_PADDLEOCR = False
    PaddleOCR = None


class MockPaddleOCR:
    """Mock/fallback OCR engine for development and testing environments."""

    def __init__(self, **kwargs):
        pass

    def ocr(self, image: np.ndarray, cls: bool = True):
        """Simulate structured OCR text detection & recognition."""
        return [
            [
                [[[50, 30], [200, 30], [200, 60], [50, 60]], ("INVOICE", 0.99)],
                [[[50, 80], [250, 80], [250, 100], [50, 100]], ("Invoice #: INV-2024-001", 0.98)],
                [[[50, 110], [200, 110], [200, 130], [50, 130]], ("Date: 2024-01-15", 0.98)],
                [[[50, 140], [220, 140], [220, 160], [50, 160]], ("Due Date: 2024-02-15", 0.97)],
                [[[50, 190], [100, 190], [100, 210], [50, 210]], ("FROM:", 0.99)],
                [[[50, 220], [250, 220], [250, 240], [50, 240]], ("Acme Corporation", 0.97)],
                [[[50, 250], [200, 250], [200, 270], [50, 270]], ("123 Main St", 0.96)],
                [[[50, 280], [250, 280], [250, 300], [50, 300]], ("New York, NY 10001", 0.96)],
                [[[50, 310], [220, 310], [220, 330], [50, 330]], ("Tax ID: 12-3456789", 0.97)],
                [[[500, 440], [600, 440], [600, 460], [500, 460]], ("Subtotal: $1,500.00", 0.99)],
                [[[500, 470], [600, 470], [600, 490], [500, 490]], ("Tax (10%): $150.00", 0.99)],
                [[[500, 500], [600, 500], [600, 520], [500, 520]], ("TOTAL: $1,650.00", 0.99)],
            ]
        ]


class FieldType(Enum):
    INVOICE_NUMBER = "invoice_number"
    INVOICE_DATE = "invoice_date"
    DUE_DATE = "due_date"
    VENDOR_NAME = "vendor_name"
    VENDOR_ADDRESS = "vendor_address"
    VENDOR_TAX_ID = "vendor_tax_id"
    SUBTOTAL = "subtotal"
    TAX_AMOUNT = "tax_amount"
    TOTAL_AMOUNT = "total_amount"
    CURRENCY = "currency"


@dataclass
class ExtractedField:
    field_type: FieldType
    value: str
    confidence: float
    bbox: list[list[int]] | None = None
    raw_text: str | None = None


@dataclass
class ExtractionResult:
    fields: list[ExtractedField] = field(default_factory=list)
    raw_ocr_text: str = ""
    raw_ocr_data: list = field(default_factory=list)
    processing_time_ms: float = 0.0
    image_shape: tuple[int, int] = (0, 0)

    def to_dict(self) -> dict:
        return {
            "fields": [
                {
                    "field_type": f.field_type.value,
                    "value": f.value,
                    "confidence": f.confidence,
                    "bbox": f.bbox,
                }
                for f in self.fields
            ],
            "raw_ocr_text": self.raw_ocr_text,
            "processing_time_ms": self.processing_time_ms,
            "image_shape": self.image_shape,
        }

    def get_field(self, field_type: FieldType) -> ExtractedField | None:
        for f in self.fields:
            if f.field_type == field_type:
                return f
        return None


class RegexPatterns:
    """Compiled regex patterns for field extraction."""

    # Use either (?i) flag OR re.IGNORECASE, not both
    INVOICE_NUMBER = [
        re.compile(
            r"(?i)\b(?:invoice|inv|bill)\s*(?:[#:]|no\.?|num(?:ber)?)\s*[:#\-\s]*([A-Z0-9\-]+)"
        ),
        re.compile(r"(?i)\b(?:invoice|inv|bill)\s*[:#\-]+\s*([A-Z0-9\-]+)"),
        re.compile(r"(?i)\b(INV|INVOICE|BILL)[\-\s]?(\d{4,})[\-\s]?(\d+)\b"),
    ]

    DATE_PATTERNS = [
        re.compile(r"\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b"),
        re.compile(r"\b\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}\b"),
        re.compile(
            r"(?i)\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b"
        ),
        re.compile(
            r"(?i)\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b"
        ),
    ]

    VENDOR_NAME = [
        re.compile(r"(?i)^(.+?)(?:\n|$)"),  # First line heuristic
        re.compile(r"(?i)(?:from|vendor|supplier|seller)[:\s]+(.+?)(?:\n|$)"),
    ]

    VENDOR_ADDRESS = [
        re.compile(r"(?i)(?:address|addr)[:\s]+(.+?)(?:\n\n|\n[A-Z]{2}\s+\d{5}|$)"),
    ]

    TAX_ID = [
        re.compile(r"(?i)(?:tax\s*id|ein|vat|gst)[:\s#]*([A-Z0-9\-]+)"),
        re.compile(r"\b\d{2}[\-\s]?\d{7}\b"),  # US EIN
        re.compile(r"\b[A-Z]{2}\d{8,12}\b"),  # EU VAT
    ]

    AMOUNT_PATTERNS = {
        FieldType.SUBTOTAL: [
            re.compile(r"(?i)\b(?:subtotal|sub\s*total|net\s*amount)\b[:\s\$€£]*([\d,]+\.?\d*)"),
        ],
        FieldType.TAX_AMOUNT: [
            re.compile(r"(?i)\b(?:tax|vat|gst|sales\s*tax)\b[^$\d\n]*[:\s\$€£]*([\d,]+\.?\d*)"),
        ],
        FieldType.TOTAL_AMOUNT: [
            re.compile(
                r"(?i)\b(?:grand\s*total|amount\s*due|balance\s*due|total)\b[:\s\$€£]*([\d,]+\.?\d*)"
            ),
        ],
    }

    CURRENCY = [
        re.compile(r"[\$\€\£\¥]\s*([\d,]+\.?\d*)"),
        re.compile(r"(?i)(?:USD|EUR|GBP|CAD|AUD)\s*([\d,]+\.?\d*)"),
    ]


class BaselineExtractor:
    """OCR + Regex baseline extractor."""

    def __init__(
        self,
        lang: str = "en",
        use_gpu: bool = False,
        det_model_dir: str | None = None,
        rec_model_dir: str | None = None,
    ):
        if HAS_PADDLEOCR:
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=lang,
                use_gpu=use_gpu,
                det_model_dir=det_model_dir,
                rec_model_dir=rec_model_dir,
                show_log=False,
            )
        else:
            self.ocr = MockPaddleOCR()
        self.patterns = RegexPatterns()

    def extract(self, image: np.ndarray) -> ExtractionResult:
        """Extract fields from image using OCR + Regex."""
        start = time.perf_counter()

        h, w = image.shape[:2]
        result = ExtractionResult(image_shape=(h, w))

        # Run OCR
        ocr_result = self.ocr.ocr(image, cls=True)
        result.raw_ocr_data = ocr_result

        # Flatten OCR results
        text_lines = []
        all_text = []
        for page in ocr_result:
            if page:
                for line in page:
                    bbox, (text, conf) = line
                    text_lines.append((text, conf, bbox))
                    all_text.append(text)

        result.raw_ocr_text = "\n".join(all_text)

        # Extract fields using regex
        full_text = result.raw_ocr_text

        # Invoice Number
        inv_num = self._extract_invoice_number(full_text, text_lines)
        if inv_num:
            result.fields.append(inv_num)

        # Dates
        dates = self._extract_dates(full_text, text_lines)
        result.fields.extend(dates)

        # Vendor
        vendor = self._extract_vendor(full_text, text_lines)
        if vendor:
            result.fields.append(vendor)

        vendor_addr = self._extract_vendor_address(full_text, text_lines)
        if vendor_addr:
            result.fields.append(vendor_addr)

        # Tax ID
        tax_id = self._extract_tax_id(full_text, text_lines)
        if tax_id:
            result.fields.append(tax_id)

        # Amounts
        amounts = self._extract_amounts(full_text, text_lines)
        result.fields.extend(amounts)

        # Currency
        currency = self._extract_currency(full_text, text_lines)
        if currency:
            result.fields.append(currency)

        result.processing_time_ms = (time.perf_counter() - start) * 1000
        return result

    def _safe_group(self, match: re.Match, group: int = 1) -> str:
        """Safely extract regex group, fallback to group(0) if not present."""
        if match.groups() and len(match.groups()) >= group:
            return match.group(group)
        return match.group(0)

    def _extract_invoice_number(self, text: str, lines: list) -> ExtractedField | None:
        for pattern in self.patterns.INVOICE_NUMBER:
            match = pattern.search(text)
            if match:
                value = self._safe_group(match, 1)
                bbox = self._find_bbox_for_text(value, lines)
                return ExtractedField(
                    field_type=FieldType.INVOICE_NUMBER,
                    value=value.strip(),
                    confidence=0.7,
                    bbox=bbox,
                    raw_text=match.group(0),
                )
        return None

    def _extract_dates(self, text: str, lines: list) -> list[ExtractedField]:
        fields = []
        found_dates = set()

        for pattern in self.patterns.DATE_PATTERNS:
            for match in pattern.finditer(text):
                value = match.group(0)
                if value in found_dates:
                    continue
                found_dates.add(value)

                # Determine if invoice date or due date based on context
                context_start = max(0, match.start() - 50)
                context = text[context_start : match.start()].lower()

                if any(kw in context for kw in ["due", "pay by", "payment"]):
                    field_type = FieldType.DUE_DATE
                else:
                    field_type = FieldType.INVOICE_DATE

                bbox = self._find_bbox_for_text(value, lines)
                fields.append(
                    ExtractedField(
                        field_type=field_type,
                        value=value,
                        confidence=0.75,
                        bbox=bbox,
                        raw_text=match.group(0),
                    )
                )

        return fields

    def _extract_vendor(self, text: str, lines: list) -> ExtractedField | None:
        # Heuristic: first meaningful line is often vendor name
        for line_text, conf, bbox in lines[:5]:
            if len(line_text.strip()) > 3 and not self._is_likely_field_label(line_text):
                return ExtractedField(
                    field_type=FieldType.VENDOR_NAME,
                    value=line_text.strip(),
                    confidence=min(conf * 0.6, 0.8),
                    bbox=bbox,
                    raw_text=line_text,
                )
        return None

    def _extract_vendor_address(self, text: str, lines: list) -> ExtractedField | None:
        for pattern in self.patterns.VENDOR_ADDRESS:
            match = pattern.search(text)
            if match:
                value = self._safe_group(match, 1).strip()
                bbox = self._find_bbox_for_text(value, lines)
                return ExtractedField(
                    field_type=FieldType.VENDOR_ADDRESS,
                    value=value,
                    confidence=0.5,
                    bbox=bbox,
                    raw_text=match.group(0),
                )
        return None

    def _extract_tax_id(self, text: str, lines: list) -> ExtractedField | None:
        for pattern in self.patterns.TAX_ID:
            match = pattern.search(text)
            if match:
                value = self._safe_group(match, 1).strip()
                bbox = self._find_bbox_for_text(value, lines)
                return ExtractedField(
                    field_type=FieldType.VENDOR_TAX_ID,
                    value=value,
                    confidence=0.65,
                    bbox=bbox,
                    raw_text=match.group(0),
                )
        return None

    def _extract_amounts(self, text: str, lines: list) -> list[ExtractedField]:
        fields = []
        for field_type, patterns in self.patterns.AMOUNT_PATTERNS.items():
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    value = self._safe_group(match, 1).replace(",", "")
                    try:
                        float(value)
                        bbox = self._find_bbox_for_text(match.group(0), lines)
                        fields.append(
                            ExtractedField(
                                field_type=field_type,
                                value=value,
                                confidence=0.7,
                                bbox=bbox,
                                raw_text=match.group(0),
                            )
                        )
                        break
                    except ValueError:
                        continue
        return fields

    def _extract_currency(self, text: str, lines: list) -> ExtractedField | None:
        for pattern in self.patterns.CURRENCY:
            match = pattern.search(text)
            if match:
                value = self._safe_group(match, 1).replace(",", "")
                try:
                    float(value)
                    # Determine currency symbol
                    symbol = match.group(0)[0]
                    currency_map = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}
                    currency = currency_map.get(symbol, "USD")
                    bbox = self._find_bbox_for_text(match.group(0), lines)
                    return ExtractedField(
                        field_type=FieldType.CURRENCY,
                        value=currency,
                        confidence=0.8,
                        bbox=bbox,
                        raw_text=match.group(0),
                    )
                except ValueError:
                    continue
        return None

    def _find_bbox_for_text(self, search_text: str, lines: list) -> list[list[int]] | None:
        """Find bounding box for text in OCR results."""
        search_clean = search_text.strip().lower()
        for line_text, conf, bbox in lines:
            if search_clean in line_text.strip().lower():
                # Convert bbox to [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] format
                return [[int(p[0]), int(p[1])] for p in bbox]
        return None

    def _is_likely_field_label(self, text: str) -> bool:
        labels = [
            "invoice",
            "date",
            "total",
            "tax",
            "amount",
            "due",
            "bill",
            "ship",
            "bill to",
            "ship to",
        ]
        text_lower = text.lower()
        return any(label in text_lower for label in labels)


# Module-level singleton for model reuse
_extractor_instance: BaselineExtractor | None = None


def create_extractor(lang: str = "en", use_gpu: bool = False) -> BaselineExtractor:
    """Factory function to create baseline extractor."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = BaselineExtractor(lang=lang, use_gpu=use_gpu)
    return _extractor_instance


def get_extractor() -> BaselineExtractor:
    """Get or create the singleton extractor instance."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = BaselineExtractor()
    return _extractor_instance
