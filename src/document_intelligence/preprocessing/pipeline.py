"""Image preprocessing pipeline for document images."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import fitz  # PyMuPDF

    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    from pdf2image import convert_from_bytes

    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False


class PreprocessStage(Enum):
    GRAYSCALE = "grayscale"
    CLAHE = "clahe"
    DESKEW = "deskew"
    DENOISE = "denoise"
    BINARIZE = "binarize"
    MORPHOLOGY = "morphology"
    LINE_REMOVAL = "line_removal"


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing pipeline."""

    # Grayscale + CLAHE
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)

    # Deskewing
    apply_deskew: bool = True
    deskew_min_angle: float = -45.0
    deskew_max_angle: float = 45.0

    # Denoising
    apply_denoise: bool = True
    denoise_h: float = 10.0
    denoise_template_window: int = 7
    denoise_search_window: int = 21

    # Binarization
    apply_binarize: bool = True
    binarize_method: str = "adaptive_gaussian"  # adaptive_gaussian, adaptive_mean, otsu
    adaptive_block_size: int = 15
    adaptive_c: float = 5.0

    # Morphology
    apply_morphology: bool = True
    morph_kernel_size: tuple[int, int] = (2, 2)
    morph_iterations: int = 1

    # Line removal
    apply_line_removal: bool = True
    horizontal_kernel_ratio: float = 1 / 30
    vertical_kernel_ratio: float = 1 / 30
    line_min_length_ratio: float = 0.5

    # Debug
    save_intermediate: bool = False
    output_dir: str | None = None


@dataclass
class PreprocessResult:
    """Result of preprocessing pipeline."""

    image: np.ndarray
    stages_applied: list[str] = field(default_factory=list)
    intermediate_images: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def save_intermediates(self, output_dir: Path) -> None:
        """Save intermediate images for debugging."""
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, img in self.intermediate_images.items():
            cv2.imwrite(str(output_dir / f"{name}.png"), img)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def apply_clahe(
    image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def deskew(
    image: np.ndarray, min_angle: float = -45.0, max_angle: float = 45.0
) -> tuple[np.ndarray, float]:
    """Deskew image using Hough Line Transform."""
    # Edge detection
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None or len(lines) == 0:
        return image, 0.0

    # Calculate angles
    angles = []
    for line in lines:
        coords = line.reshape(-1)
        if len(coords) >= 4:
            x1, y1, x2, y2 = coords[:4]
            angle = np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1)))
            if min_angle <= angle <= max_angle:
                angles.append(angle)

    if not angles:
        return image, 0.0

    # Use median angle to avoid outliers
    median_angle = np.median(angles)

    # Skip if angle is very small
    if abs(median_angle) < 0.5:
        return image, 0.0

    # Rotate image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)

    # Calculate new bounds
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(
        image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    return rotated, median_angle


def denoise(
    image: np.ndarray, h: float = 10.0, template_window: int = 7, search_window: int = 21
) -> np.ndarray:
    """Apply Non-local Means Denoising."""
    return cv2.fastNlMeansDenoising(image, None, h, template_window, search_window)


def binarize(
    image: np.ndarray, method: str = "adaptive_gaussian", block_size: int = 15, c: float = 5.0
) -> np.ndarray:
    """Binarize image using various methods."""
    if method == "adaptive_gaussian":
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
        )
    if method == "adaptive_mean":
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c
        )
    if method == "otsu":
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    raise ValueError(f"Unknown binarization method: {method}")


def morphology(
    image: np.ndarray, kernel_size: tuple[int, int] = (2, 2), iterations: int = 1
) -> np.ndarray:
    """Apply morphological operations to clean up binary image."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    # Close to fill small gaps, then open to remove noise
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return opened


def remove_lines(
    image: np.ndarray,
    horizontal_ratio: float = 1 / 30,
    vertical_ratio: float = 1 / 30,
    min_length_ratio: float = 0.5,
) -> np.ndarray:
    """Remove horizontal and vertical lines (table borders)."""
    result = image.copy()
    h, w = image.shape[:2]

    # Horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * horizontal_ratio), 1))
    horizontal_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=1)

    # Vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h * vertical_ratio)))
    vertical_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=1)

    # Combine and remove
    lines = cv2.bitwise_or(horizontal_lines, vertical_lines)
    result = cv2.subtract(result, lines)

    return result


class PreprocessingPipeline:
    """Configurable preprocessing pipeline for document images."""

    def __init__(self, config: PreprocessConfig | None = None):
        self.config = config or PreprocessConfig()

    def process(self, image: np.ndarray) -> PreprocessResult:
        """Run full preprocessing pipeline."""
        result = PreprocessResult(image=image.copy())
        current = image.copy()

        # Stage 1: Grayscale
        if len(current.shape) == 3:
            current = to_grayscale(current)
            result.stages_applied.append(PreprocessStage.GRAYSCALE.value)
            result.intermediate_images["01_grayscale"] = current.copy()

        # Stage 2: CLAHE
        if self.config.apply_clahe:
            current = apply_clahe(
                current, self.config.clahe_clip_limit, self.config.clahe_tile_grid_size
            )
            result.stages_applied.append(PreprocessStage.CLAHE.value)
            result.intermediate_images["02_clahe"] = current.copy()

        # Stage 3: Deskew
        if self.config.apply_deskew:
            current, angle = deskew(
                current, self.config.deskew_min_angle, self.config.deskew_max_angle
            )
            result.stages_applied.append(PreprocessStage.DESKEW.value)
            result.metadata["deskew_angle"] = angle
            result.intermediate_images["03_deskew"] = current.copy()

        # Stage 4: Denoise
        if self.config.apply_denoise:
            current = denoise(
                current,
                self.config.denoise_h,
                self.config.denoise_template_window,
                self.config.denoise_search_window,
            )
            result.stages_applied.append(PreprocessStage.DENOISE.value)
            result.intermediate_images["04_denoise"] = current.copy()

        # Stage 5: Binarize
        if self.config.apply_binarize:
            current = binarize(
                current,
                self.config.binarize_method,
                self.config.adaptive_block_size,
                self.config.adaptive_c,
            )
            result.stages_applied.append(PreprocessStage.BINARIZE.value)
            result.intermediate_images["05_binarize"] = current.copy()

        # Stage 6: Morphology
        if self.config.apply_morphology:
            current = morphology(
                current, self.config.morph_kernel_size, self.config.morph_iterations
            )
            result.stages_applied.append(PreprocessStage.MORPHOLOGY.value)
            result.intermediate_images["06_morphology"] = current.copy()

        # Stage 7: Line removal
        if self.config.apply_line_removal:
            current = remove_lines(
                current,
                self.config.horizontal_kernel_ratio,
                self.config.vertical_kernel_ratio,
                self.config.line_min_length_ratio,
            )
            result.stages_applied.append(PreprocessStage.LINE_REMOVAL.value)
            result.intermediate_images["07_line_removal"] = current.copy()

        result.image = current

        # Save intermediates if configured
        if self.config.save_intermediate and self.config.output_dir:
            result.save_intermediates(Path(self.config.output_dir))

        return result

    def process_file(self, input_path: str, output_path: str | None = None) -> PreprocessResult:
        """Process image file and optionally save result."""
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not read image: {input_path}")

        result = self.process(image)

        if output_path:
            cv2.imwrite(output_path, result.image)

        return result


def create_pipeline(
    clahe: bool = True,
    deskew: bool = True,
    denoise: bool = True,
    binarize: bool = True,
    morphology: bool = True,
    line_removal: bool = True,
    save_intermediate: bool = False,
    output_dir: str | None = None,
) -> PreprocessingPipeline:
    """Factory function to create preprocessing pipeline with common settings."""
    config = PreprocessConfig(
        apply_clahe=clahe,
        apply_deskew=deskew,
        apply_denoise=denoise,
        apply_binarize=binarize,
        apply_morphology=morphology,
        apply_line_removal=line_removal,
        save_intermediate=save_intermediate,
        output_dir=output_dir,
    )
    return PreprocessingPipeline(config)


# PDF and multi-page support


@dataclass
class DocumentPage:
    """Represents a single page of a document."""

    page_number: int
    image: np.ndarray
    width: int
    height: int


def pdf_to_images(file_bytes: bytes, dpi: int = 200) -> list[DocumentPage]:
    """
    Convert PDF to list of images.

    Uses PyMuPDF (fitz) as primary, falls back to pdf2image.
    """
    pages = []

    if HAS_PYMUPDF:
        return _pdf_to_images_pymupdf(file_bytes, dpi)
    if HAS_PDF2IMAGE:
        return _pdf_to_images_pdf2image(file_bytes, dpi)
    raise RuntimeError(
        "PDF support requires PyMuPDF or pdf2image. Install with: pip install pymupdf"
    )


def _pdf_to_images_pymupdf(file_bytes: bytes, dpi: int = 200) -> list[DocumentPage]:
    """Convert PDF using PyMuPDF."""
    pages = []
    doc = fitz.open(stream=file_bytes, filetype="pdf")

    zoom = dpi / 72.0  # 72 DPI is default
    mat = fitz.Matrix(zoom, zoom)

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        pages.append(
            DocumentPage(
                page_number=page_num + 1,
                image=img_bgr,
                width=pix.width,
                height=pix.height,
            )
        )

    doc.close()
    return pages


def _pdf_to_images_pdf2image(file_bytes: bytes, dpi: int = 200) -> list[DocumentPage]:
    """Convert PDF using pdf2image."""
    pages = []
    images = convert_from_bytes(file_bytes, dpi=dpi)

    for page_num, img in enumerate(images):
        img_array = np.array(img)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        pages.append(
            DocumentPage(
                page_number=page_num + 1,
                image=img_bgr,
                width=img.width,
                height=img.height,
            )
        )

    return pages


def process_document_pages(
    file_bytes: bytes,
    mime_type: str,
    pipeline: PreprocessingPipeline | None = None,
    dpi: int = 200,
) -> list[PreprocessResult]:
    """
    Process all pages of a document (PDF or image).

    Returns list of PreprocessResult, one per page.
    """
    pipeline = pipeline or create_pipeline()
    results = []

    if mime_type == "application/pdf":
        pages = pdf_to_images(file_bytes, dpi)
    else:
        # Single image
        image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image")
        pages = [
            DocumentPage(page_number=1, image=image, width=image.shape[1], height=image.shape[0])
        ]

    for page in pages:
        result = pipeline.process(page.image)
        result.metadata["page_number"] = page.page_number
        result.metadata["original_width"] = page.width
        result.metadata["original_height"] = page.height
        results.append(result)

    return results


def merge_page_results(page_results: list[PreprocessResult]) -> PreprocessResult:
    """
    Merge multi-page results into a single result.

    For now, concatenates OCR text and merges metadata.
    """
    if not page_results:
        raise ValueError("No page results to merge")

    if len(page_results) == 1:
        return page_results[0]

    # Merge metadata
    merged_metadata = page_results[0].metadata.copy()
    merged_metadata["page_count"] = len(page_results)
    merged_metadata["pages"] = [
        {"page_number": r.metadata.get("page_number"), "stages": r.stages_applied}
        for r in page_results
    ]

    # Use first page's image as representative (for display)
    merged_image = page_results[0].image

    # Combine all stages
    all_stages = []
    for r in page_results:
        all_stages.extend(r.stages_applied)

    return PreprocessResult(
        image=merged_image,
        stages_applied=list(dict.fromkeys(all_stages)),  # Unique preserving order
        intermediate_images={},  # Don't keep all intermediates
        metadata=merged_metadata,
    )
