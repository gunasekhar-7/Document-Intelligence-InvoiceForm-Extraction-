"""LayoutLMv3 fine-tuning training pipeline."""

import json
from dataclasses import dataclass

import mlflow
import mlflow.pytorch
import torch
from torch.utils.data import Dataset
from transformers import (
    EarlyStoppingCallback,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    Trainer,
    TrainingArguments,
)

from document_intelligence.config import settings
from document_intelligence.logging import get_logger

logger = get_logger(__name__)


# Entity labels (BIO scheme)
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

LABEL2ID = {label: i for i, label in enumerate(ENTITY_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(ENTITY_LABELS)}


@dataclass
class TrainingConfig:
    """Configuration for LayoutLMv3 fine-tuning."""

    model_name: str = "microsoft/layoutlmv3-base"
    output_dir: str = "models/layoutlmv3-finetuned"
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    fp16: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 200
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_f1"
    greater_is_better: bool = True
    early_stopping_patience: int = 3
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    report_to: str = "mlflow"
    run_name: str = "layoutlmv3-finetune"
    seed: int = 42


def normalize_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    """Normalize bbox to 0-1000 scale as expected by LayoutLMv3."""
    return [
        int(1000 * bbox[0] / width),
        int(1000 * bbox[1] / height),
        int(1000 * bbox[2] / width),
        int(1000 * bbox[3] / height),
    ]


def align_labels_with_tokens(
    word_labels: list[int],
    word_ids: list[int | None],
) -> list[int]:
    """
    Align word-level labels to token-level labels.

    For subword tokens, assign the label of the first subword (B- tag)
    and I- tag for subsequent subwords of the same word.
    """
    aligned_labels = []
    previous_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            # Special token
            aligned_labels.append(-100)
        elif word_idx != previous_word_idx:
            # First subword of a word
            aligned_labels.append(word_labels[word_idx])
        else:
            # Subsequent subword - convert B- to I- if needed
            label = word_labels[word_idx]
            if label % 2 == 1:  # B- tag (odd indices are B- tags)
                aligned_labels.append(label + 1)  # Convert to I- tag
            else:
                aligned_labels.append(label)
        previous_word_idx = word_idx

    return aligned_labels


class InvoiceDataset(Dataset):
    """Dataset for LayoutLMv3 token classification with proper token-bbox alignment."""

    def __init__(
        self,
        data: list[dict],
        processor: LayoutLMv3Processor,
        max_length: int = 512,
    ):
        self.data = data
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load image
        from PIL import Image

        image = Image.open(item["image_path"]).convert("RGB")
        width, height = image.size

        # Get word-level data
        words = item["tokens"]
        word_boxes = item["bboxes"]
        word_labels = item["ner_tags"]

        # Normalize bboxes to 0-1000 scale
        normalized_boxes = [normalize_bbox(box, width, height) for box in word_boxes]

        # Tokenize with processor (this handles image + text tokenization)
        encoding = self.processor(
            image,
            words,
            boxes=normalized_boxes,
            word_labels=word_labels,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        # The processor returns word_ids that map tokens to words
        word_ids = encoding.word_ids()

        # Align labels to token level
        if word_ids is not None:
            aligned_labels = align_labels_with_tokens(word_labels, word_ids)
            encoding["labels"] = torch.tensor(aligned_labels)

        # Remove batch dimension
        return {k: v.squeeze(0) for k, v in encoding.items()}


def load_dataset_from_json(json_path: str) -> list[dict]:
    """Load dataset from JSON annotation files."""
    with open(json_path) as f:
        data = json.load(f)

    processed = []
    for item in data:
        # Validate required fields
        required = ["image_path", "tokens", "bboxes", "ner_tags"]
        if not all(k in item for k in required):
            logger.warning(
                f"Skipping item with missing fields: {item.get('image_path', 'unknown')}"
            )
            continue

        # Validate lengths match
        if not (len(item["tokens"]) == len(item["bboxes"]) == len(item["ner_tags"])):
            logger.warning(f"Length mismatch in item: {item.get('image_path', 'unknown')}")
            continue

        processed.append(
            {
                "image_path": item["image_path"],
                "tokens": item["tokens"],
                "bboxes": item["bboxes"],
                "ner_tags": item["ner_tags"],
            }
        )

    logger.info(f"Loaded {len(processed)} samples from {json_path}")
    return processed


def compute_metrics(eval_pred):
    """Compute metrics for token classification."""
    from seqeval.metrics import f1_score, precision_score, recall_score

    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)

    # Remove ignored index (special tokens)
    true_predictions = [
        [ID2LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [ID2LABEL[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return {
        "f1": f1_score(true_labels, true_predictions),
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
    }


def train_layoutlmv3(
    train_data_path: str,
    val_data_path: str,
    config: TrainingConfig | None = None,
    model_path: str | None = None,
) -> LayoutLMv3ForTokenClassification:
    """Train LayoutLMv3 model."""
    config = config or TrainingConfig()

    # Set seed
    torch.manual_seed(config.seed)

    # Initialize MLflow
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("layoutlmv3-invoice-extraction")

    with mlflow.start_run(run_name=config.run_name):
        # Log parameters
        mlflow.log_params(config.__dict__)

        # Load processor
        processor = LayoutLMv3Processor.from_pretrained(config.model_name, apply_ocr=False)

        # Load datasets
        train_data = load_dataset_from_json(train_data_path)
        val_data = load_dataset_from_json(val_data_path)

        train_dataset = InvoiceDataset(train_data, processor)
        val_dataset = InvoiceDataset(val_data, processor)

        # Load model
        if model_path:
            model = LayoutLMv3ForTokenClassification.from_pretrained(
                model_path,
                num_labels=len(ENTITY_LABELS),
                id2label=ID2LABEL,
                label2id=LABEL2ID,
            )
        else:
            model = LayoutLMv3ForTokenClassification.from_pretrained(
                config.model_name,
                num_labels=len(ENTITY_LABELS),
                id2label=ID2LABEL,
                label2id=LABEL2ID,
            )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            max_grad_norm=config.max_grad_norm,
            fp16=config.fp16,
            gradient_checkpointing=config.gradient_checkpointing,
            logging_steps=config.logging_steps,
            eval_steps=config.eval_steps,
            save_steps=config.save_steps,
            eval_strategy=config.eval_strategy,
            save_strategy=config.save_strategy,
            load_best_model_at_end=config.load_best_model_at_end,
            metric_for_best_model=config.metric_for_best_model,
            greater_is_better=config.greater_is_better,
            dataloader_num_workers=config.dataloader_num_workers,
            remove_unused_columns=config.remove_unused_columns,
            report_to=config.report_to,
            run_name=config.run_name,
            seed=config.seed,
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=processor.tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(config.early_stopping_patience)],
        )

        # Train
        logger.info("Starting training")
        trainer.train()

        # Save best model
        trainer.save_model(config.output_dir)
        processor.save_pretrained(config.output_dir)

        # Log model
        mlflow.pytorch.log_model(model, "model")

        logger.info("Training completed", output_dir=config.output_dir)
        return model


def export_to_onnx(model_path: str, output_path: str):
    """Export model to ONNX format for production inference."""
    import torch
    from transformers import LayoutLMv3Processor

    model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
    processor = LayoutLMv3Processor.from_pretrained(model_path, apply_ocr=False)

    model.eval()

    # Dummy inputs
    dummy_input = {
        "input_ids": torch.randint(0, 30000, (1, 512)),
        "attention_mask": torch.ones(1, 512),
        "bbox": torch.randint(0, 1000, (1, 512, 4)),
        "pixel_values": torch.randn(1, 3, 224, 224),
    }

    torch.onnx.export(
        model,
        (
            dummy_input["input_ids"],
            dummy_input["attention_mask"],
            dummy_input["bbox"],
            dummy_input["pixel_values"],
        ),
        output_path,
        input_names=["input_ids", "attention_mask", "bbox", "pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "bbox": {0: "batch", 1: "sequence"},
            "pixel_values": {0: "batch"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=14,
    )

    logger.info("Model exported to ONNX", path=output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train LayoutLMv3 for invoice extraction")
    parser.add_argument("--train-data", type=str, required=True, help="Path to training data JSON")
    parser.add_argument("--val-data", type=str, required=True, help="Path to validation data JSON")
    parser.add_argument("--output-dir", type=str, default="models/layoutlmv3-finetuned")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument(
        "--model-path", type=str, help="Path to pretrained model for continued training"
    )
    args = parser.parse_args()

    config = TrainingConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    train_layoutlmv3(args.train_data, args.val_data, config, args.model_path)
