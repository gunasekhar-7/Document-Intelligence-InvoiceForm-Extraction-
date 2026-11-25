"""Synthetic invoice generator with exact LayoutLMv3 token, bounding box, and BIO tag annotations."""

import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from faker import Faker
from PIL import Image, ImageDraw, ImageFont

fake = Faker()

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


@dataclass
class LineItem:
    description: str
    quantity: float
    unit_price: float
    total: float


@dataclass
class InvoiceData:
    invoice_number: str
    invoice_date: str
    due_date: str
    vendor_name: str
    vendor_address: str
    vendor_tax_id: str
    currency: str
    line_items: list[LineItem]
    subtotal: float
    tax_rate: float
    tax_amount: float
    total_amount: float
    notes: str = ""


class InvoiceTemplate:
    """Invoice template with synchronized token and bounding-box tracking."""

    def __init__(self, width: int = 800, height: int = 1000):
        self.width = width
        self.height = height
        self.margin = 50
        self.line_height = 25
        self.font_size = 14
        self.title_font_size = 24
        self.header_font_size = 16

    def render_with_tokens(
        self, data: InvoiceData
    ) -> tuple[Image.Image, list[str], list[list[int]], list[int]]:
        img = Image.new("RGB", (self.width, self.height), "white")
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", self.font_size)
            title_font = ImageFont.truetype("arial.ttf", self.title_font_size)
            header_font = ImageFont.truetype("arial.ttf", self.header_font_size)
        except Exception:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
            header_font = ImageFont.load_default()

        tokens: list[str] = []
        bboxes: list[list[int]] = []
        ner_tags: list[int] = []

        def draw_token_span(text: str, x: int, y: int, f, tag_prefix: str | None = None):
            words = text.strip().split()
            cur_x = x
            for idx, word in enumerate(words):
                bbox = draw.textbbox((cur_x, y), word, font=f)
                draw.text((cur_x, y), word, fill="black", font=f)
                tokens.append(word)
                bboxes.append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
                if tag_prefix:
                    tag_name = f"B-{tag_prefix}" if idx == 0 else f"I-{tag_prefix}"
                    ner_tags.append(LABEL2ID.get(tag_name, 0))
                else:
                    ner_tags.append(0)
                cur_x = bbox[2] + 6

        y = self.margin

        # Title
        draw_token_span("INVOICE", self.margin, y, title_font, tag_prefix=None)
        y += self.title_font_size + 15

        # Invoice number & Dates
        draw_token_span("Invoice #:", self.margin, y, font, tag_prefix=None)
        draw_token_span(data.invoice_number, self.margin + 90, y, font, tag_prefix="INVOICE_NUM")

        draw_token_span("Date:", self.width - 250, y, font, tag_prefix=None)
        draw_token_span(data.invoice_date, self.width - 190, y, font, tag_prefix="INVOICE_DATE")
        y += self.line_height

        draw_token_span("Due Date:", self.width - 250, y, font, tag_prefix=None)
        draw_token_span(data.due_date, self.width - 165, y, font, tag_prefix="DUE_DATE")
        y += self.line_height * 2

        # Vendor info
        draw_token_span("FROM:", self.margin, y, header_font, tag_prefix=None)
        y += self.header_font_size + 5
        draw_token_span(data.vendor_name, self.margin, y, font, tag_prefix="VENDOR_NAME")
        y += self.line_height

        for addr_line in data.vendor_address.split("\n"):
            draw_token_span(addr_line, self.margin, y, font, tag_prefix="VENDOR_ADDRESS")
            y += self.line_height

        draw_token_span("Tax ID:", self.margin, y, font, tag_prefix=None)
        draw_token_span(data.vendor_tax_id, self.margin + 60, y, font, tag_prefix="VENDOR_TAX_ID")
        y += self.line_height * 2

        # Line items header
        draw_token_span("DESCRIPTION", self.margin, y, header_font, tag_prefix=None)
        draw_token_span("QTY", self.margin + 350, y, header_font, tag_prefix=None)
        draw_token_span("UNIT PRICE", self.margin + 440, y, header_font, tag_prefix=None)
        draw_token_span("TOTAL", self.margin + 590, y, header_font, tag_prefix=None)
        y += self.header_font_size + 6

        # Line items
        for item in data.line_items:
            draw_token_span(
                item.description[:40], self.margin, y, font, tag_prefix="LINE_ITEM_DESC"
            )
            draw_token_span(
                str(item.quantity), self.margin + 350, y, font, tag_prefix="LINE_ITEM_QTY"
            )
            draw_token_span(
                f"{item.unit_price:.2f}", self.margin + 440, y, font, tag_prefix="LINE_ITEM_PRICE"
            )
            draw_token_span(
                f"{item.total:.2f}", self.margin + 590, y, font, tag_prefix="LINE_ITEM_TOTAL"
            )
            y += self.line_height

        y += self.line_height

        # Totals
        draw_token_span("Subtotal:", self.margin + 440, y, font, tag_prefix=None)
        draw_token_span(data.currency, self.margin + 550, y, font, tag_prefix="CURRENCY")
        draw_token_span(f"{data.subtotal:.2f}", self.margin + 590, y, font, tag_prefix="SUBTOTAL")
        y += self.line_height

        draw_token_span(
            f"Tax ({data.tax_rate * 100:.1f}%):", self.margin + 440, y, font, tag_prefix=None
        )
        draw_token_span(
            f"{data.tax_amount:.2f}", self.margin + 590, y, font, tag_prefix="TAX_AMOUNT"
        )
        y += self.line_height

        draw_token_span("TOTAL:", self.margin + 440, y, header_font, tag_prefix=None)
        draw_token_span(data.currency, self.margin + 550, y, header_font, tag_prefix="CURRENCY")
        draw_token_span(
            f"{data.total_amount:.2f}", self.margin + 590, y, header_font, tag_prefix="TOTAL_AMOUNT"
        )

        return img, tokens, bboxes, ner_tags


def generate_invoice_number() -> str:
    prefixes = ["INV", "INVOICE", "BILL", "REC"]
    prefix = random.choice(prefixes)
    year = datetime.now().year
    num = random.randint(1000, 99999)
    return f"{prefix}-{year}-{num:05d}"


def generate_vendor_data() -> tuple[str, str, str]:
    name = fake.company()
    address = f"{fake.street_address()}\n{fake.city()}, {fake.state_abbr()} {fake.zipcode()}"
    tax_id = f"{random.randint(10, 99)}-{random.randint(1000000, 9999999)}"
    return name, address, tax_id


def generate_line_items(count: int | None = None) -> list[LineItem]:
    if count is None:
        count = random.randint(1, 5)
    items = []
    descriptions = [
        "Consulting Services",
        "Software License",
        "Cloud Server Hosting",
        "Technical Support",
        "Security Audit",
        "API Subscription",
    ]
    for _ in range(count):
        desc = random.choice(descriptions)
        qty = round(random.uniform(1, 10), 0)
        price = round(random.uniform(20, 500), 2)
        total = round(qty * price, 2)
        items.append(LineItem(desc, qty, price, total))
    return items


def generate_invoice_data() -> InvoiceData:
    invoice_number = generate_invoice_number()
    vendor_name, vendor_address, vendor_tax_id = generate_vendor_data()
    line_items = generate_line_items()
    subtotal = round(sum(item.total for item in line_items), 2)
    tax_rate = 0.10
    tax_amount = round(subtotal * tax_rate, 2)
    total_amount = round(subtotal + tax_amount, 2)

    invoice_date = fake.date_between(start_date="-1y", end_date="today").strftime("%Y-%m-%d")
    due_date = (datetime.strptime(invoice_date, "%Y-%m-%d") + timedelta(days=30)).strftime(
        "%Y-%m-%d"
    )
    currency = "USD"

    return InvoiceData(
        invoice_number=invoice_number,
        invoice_date=invoice_date,
        due_date=due_date,
        vendor_name=vendor_name,
        vendor_address=vendor_address,
        vendor_tax_id=vendor_tax_id,
        currency=currency,
        line_items=line_items,
        subtotal=subtotal,
        tax_rate=tax_rate,
        tax_amount=tax_amount,
        total_amount=total_amount,
    )


def generate_dataset(
    output_dir: str,
    count: int = 100,
    split: tuple[float, float, float] = (0.7, 0.15, 0.15),
    add_noise: bool = False,
) -> dict:
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    template = InvoiceTemplate()
    samples = []
    ground_truth_records = []

    for i in range(count):
        data = generate_invoice_data()
        img, tokens, bboxes, ner_tags = template.render_with_tokens(data)

        img_filename = f"invoice_{i:06d}.png"
        img_path = images_dir / img_filename
        img.save(img_path)

        sample = {
            "image_path": str(img_path.resolve()),
            "tokens": tokens,
            "bboxes": bboxes,
            "ner_tags": ner_tags,
        }
        samples.append(sample)

        ground_truth_records.append(
            {
                "document_id": f"doc-{i:06d}",
                "filename": img_filename,
                "fields": {
                    "invoice_number": data.invoice_number,
                    "invoice_date": data.invoice_date,
                    "due_date": data.due_date,
                    "vendor_name": data.vendor_name,
                    "vendor_address": data.vendor_address,
                    "vendor_tax_id": data.vendor_tax_id,
                    "currency": data.currency,
                    "subtotal": str(data.subtotal),
                    "tax_amount": str(data.tax_amount),
                    "total_amount": str(data.total_amount),
                },
            }
        )

    random.shuffle(samples)
    train_end = int(count * split[0])
    val_end = train_end + int(count * split[1])

    dataset = {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:],
    }

    # Save training annotations
    for split_name, split_data in dataset.items():
        with open(output_path / f"{split_name}_annotations.json", "w") as f:
            json.dump(split_data, f, indent=2)

    # Save standardized ground truth for evaluation
    with open(output_path / "ground_truth.json", "w") as f:
        json.dump(ground_truth_records, f, indent=2)

    return dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic invoice dataset")
    parser.add_argument("--output", type=str, default="data/synthetic")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument(
        "--noise", action="store_true", default=False, help="Enable visual noise augmentation"
    )
    parser.add_argument(
        "--no-noise", action="store_true", default=True, help="Disable visual noise augmentation"
    )
    args = parser.parse_args()

    add_noise = args.noise and not args.no_noise
    dataset = generate_dataset(output_dir=args.output, count=args.count, add_noise=add_noise)
    print(f"Generated {args.count} invoices:")
    for split, data in dataset.items():
        print(f"  {split}: {len(data)}")
