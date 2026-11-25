"""Model serving engine for LayoutLMv3 and ONNX inference."""

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from document_intelligence.config import settings
from document_intelligence.logging import get_logger

logger = get_logger(__name__)

# Entity labels matching training BIO scheme
ENTITY_LABELS = [
    "O",
    "B-INVOICE_NUM",
    "I-INVOICE_NUM",
    "B-INVOICE_DATE",
    "I-INVOICE_DATE",
    "B-DUE_DATE",
    "I-DUE_DATE",
    "B-VENDOR_NAME",
    "I-VENDOR_NAME",
    "B-VENDOR_ADDRESS",
    "I-VENDOR_ADDRESS",
    "B-VENDOR_TAX_ID",
    "I-VENDOR_TAX_ID",
    "B-SUBTOTAL",
    "I-SUBTOTAL",
    "B-TAX_AMOUNT",
    "I-TAX_AMOUNT",
    "B-TOTAL_AMOUNT",
    "I-TOTAL_AMOUNT",
    "B-CURRENCY",
    "I-CURRENCY",
    "B-LINE_ITEM_DESC",
    "I-LINE_ITEM_DESC",
    "B-LINE_ITEM_QTY",
    "I-LINE_ITEM_QTY",
    "B-LINE_ITEM_PRICE",
    "I-LINE_ITEM_PRICE",
    "B-LINE_ITEM_TOTAL",
    "I-LINE_ITEM_TOTAL",
]

ID2LABEL = {i: label for i, label in enumerate(ENTITY_LABELS)}
LABEL2ID = {label: i for i, label in enumerate(ENTITY_LABELS)}


class DocumentIntelligenceModelServer:
    """Production serving engine supporting ONNX and PyTorch LayoutLMv3."""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or settings.model_path
        self.processor = None
        self.model = None
        self.onnx_session = None
        self.is_loaded = False
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Attempt to load ONNX or PyTorch model from configured path."""
        model_dir = Path(self.model_path)
        onnx_file = model_dir / "model.onnx"

        # Try ONNX first for fast production inference
        if onnx_file.is_file():
            try:
                import onnxruntime as ort

                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if "gpu" in ort.get_available_providers()
                    else ["CPUExecutionProvider"]
                )
                self.onnx_session = ort.InferenceSession(str(onnx_file), providers=providers)
                from transformers import LayoutLMv3Processor

                self.processor = LayoutLMv3Processor.from_pretrained(
                    str(model_dir), apply_ocr=False
                )
                self.is_loaded = True
                logger.warning("Loaded ONNX LayoutLMv3 model", path=str(onnx_file))
                return
            except Exception as e:
                logger.warning("Failed to load ONNX model, attempting PyTorch", error=str(e))

        # Try PyTorch model directory
        if model_dir.is_dir() and (model_dir / "config.json").is_file():
            try:
                from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

                self.processor = LayoutLMv3Processor.from_pretrained(
                    str(model_dir), apply_ocr=False
                )
                self.model = LayoutLMv3ForTokenClassification.from_pretrained(str(model_dir))
                self.model.eval()
                self.is_loaded = True
                logger.info("Loaded PyTorch LayoutLMv3 model", path=str(model_dir))
                return
            except Exception as e:
                logger.warning(
                    "Failed to load PyTorch model from directory", path=str(model_dir), error=str(e)
                )

        logger.info(
            "No fine-tuned LayoutLMv3 model found at configured path. Using high-accuracy baseline extractor as primary.",
            path=self.model_path,
        )

    def predict(
        self,
        image: np.ndarray,
        words: list[str],
        boxes: list[list[int]],
    ) -> dict[str, Any]:
        """Run LayoutLMv3 inference on image with OCR tokens."""
        if not self.is_loaded or not words:
            return {"fields": {}, "confidence": {}, "entities": []}

        try:
            import torch

            pil_image = Image.fromarray(image).convert("RGB")
            w, h = pil_image.size

            # Normalize boxes to 0-1000
            norm_boxes = [
                [
                    int(1000 * max(0, min(b[0], w)) / w),
                    int(1000 * max(0, min(b[1], h)) / h),
                    int(1000 * max(0, min(b[2], w)) / w),
                    int(1000 * max(0, min(b[3], h)) / h),
                ]
                for b in boxes
            ]

            encoding = self.processor(
                pil_image,
                words,
                boxes=norm_boxes,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length",
            )

            if self.onnx_session:
                inputs = {
                    "input_ids": encoding["input_ids"].numpy(),
                    "attention_mask": encoding["attention_mask"].numpy(),
                    "bbox": encoding["bbox"].numpy(),
                    "pixel_values": encoding["pixel_values"].numpy(),
                }
                outputs = self.onnx_session.run(None, inputs)
                logits = outputs[0]
                preds = np.argmax(logits, axis=-1)[0]
                probs = np.max(
                    np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True), axis=-1
                )[0]
            else:
                with torch.no_grad():
                    outputs = self.model(**encoding)
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=-1).squeeze().tolist()
                    probs = torch.softmax(logits, dim=-1).max(dim=-1).values.squeeze().tolist()

            word_ids = encoding.word_ids(batch_index=0)
            entities = []
            extracted_fields: dict[str, str] = {}
            confidence_scores: dict[str, float] = {}

            tag_to_field = {
                "INVOICE_NUM": "invoice_number",
                "INVOICE_DATE": "invoice_date",
                "DUE_DATE": "due_date",
                "VENDOR_NAME": "vendor_name",
                "VENDOR_ADDRESS": "vendor_address",
                "VENDOR_TAX_ID": "vendor_tax_id",
                "SUBTOTAL": "subtotal",
                "TAX_AMOUNT": "tax_amount",
                "TOTAL_AMOUNT": "total_amount",
                "CURRENCY": "currency",
            }

            current_tag = None
            current_tokens = []
            current_confidences = []

            for token_idx, word_idx in enumerate(word_ids):
                if word_idx is None or word_idx >= len(words):
                    continue
                label_id = preds[token_idx]
                label_name = ID2LABEL.get(label_id, "O")
                prob = float(probs[token_idx])

                if label_name.startswith("B-"):
                    if current_tag and current_tokens:
                        tag_clean = current_tag.replace("B-", "").replace("I-", "")
                        field_key = tag_to_field.get(tag_clean)
                        if field_key:
                            extracted_fields[field_key] = " ".join(current_tokens)
                            confidence_scores[field_key] = sum(current_confidences) / len(
                                current_confidences
                            )
                    current_tag = label_name
                    current_tokens = [words[word_idx]]
                    current_confidences = [prob]
                elif label_name.startswith("I-") and current_tag:
                    current_tokens.append(words[word_idx])
                    current_confidences.append(prob)
                else:
                    if current_tag and current_tokens:
                        tag_clean = current_tag.replace("B-", "").replace("I-", "")
                        field_key = tag_to_field.get(tag_clean)
                        if field_key:
                            extracted_fields[field_key] = " ".join(current_tokens)
                            confidence_scores[field_key] = sum(current_confidences) / len(
                                current_confidences
                            )
                    current_tag = None
                    current_tokens = []
                    current_confidences = []

            return {
                "fields": extracted_fields,
                "confidence": confidence_scores,
                "entities": entities,
            }

        except Exception as e:
            logger.error("Error during LayoutLMv3 inference", error=str(e))
            return {"fields": {}, "confidence": {}, "entities": []}


_model_server_instance: DocumentIntelligenceModelServer | None = None


def get_model_server() -> DocumentIntelligenceModelServer:
    """Get or create singleton ModelServer instance."""
    global _model_server_instance
    if _model_server_instance is None:
        _model_server_instance = DocumentIntelligenceModelServer()
    return _model_server_instance
