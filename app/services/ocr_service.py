import pymupdf
import easyocr
import numpy as np
from PIL import Image
import io
import ssl
import torch
import os

from loguru import logger
from dotenv import load_dotenv

load_dotenv()

from app.services.ner_service import NerService

# Handle SSL certificate verification issues for model downloads
# NOTE: In production, it's better to ensure correct CA certs are installed.
if os.getenv("SKIP_SSL_VERIFY", "false").lower() == "true":
    ssl._create_default_https_context = ssl._create_unverified_context

from typing import Protocol, List, Dict, Optional


class OCRError(Exception):
    """Base exception for OCR related errors."""
    pass


class PDFOpenError(OCRError):
    """Raised when a PDF file cannot be opened."""
    pass


class OCRProcessingError(OCRError):
    """Raised when OCR processing fails."""
    pass


class OCRProvider(Protocol):
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        ...
    def get_word_bounding_boxes(self, pdf_path: str) -> List[Dict]:
        ...
    def find_name_bounding_boxes(self, pdf_path: str, text: str) -> List[Dict]:
        ...

class OCRService:
    def __init__(self, pdf_path: Optional[str] = None, ner_service_class=NerService, gpu: Optional[bool] = None):
        """
        Initialize OCR Service.
        
        Args:
            pdf_path: Path to the PDF file (optional, for backward compatibility).
            ner_service_class: Class to use for NER (allows injection).
            gpu: Whether to use GPU. If None, auto-detects.
        """
        self.pdf_path = pdf_path
        if gpu is None:
            # Auto-detect GPU (CUDA or MPS for Mac)
            gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
            logger.info(f"Auto-detected GPU availability: {gpu}")
        
        try:
            self.ocr_engine = easyocr.Reader(['en'], gpu=gpu)
            self.ner_service_class = ner_service_class
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            raise OCRProcessingError(f"OCR initialization failed: {e}")

    def extract_text_from_pdf(self, pdf_path: Optional[str] = None) -> str:
        """Extract text from a scanned PDF using OCR."""
        pdf_path = pdf_path or self.pdf_path
        if not pdf_path:
            raise ValueError("pdf_path must be provided")

        logger.info(f"Extracting text from PDF: {pdf_path}")
        full_text = ""

        try:
            doc = pymupdf.open(pdf_path)
            try:
                # FIX: Ensure we process all pages (some versions skip page 0)
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    pix = page.get_pixmap()
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    img_array = np.array(img)
                    
                    # EasyOCR returns a list of results: (bbox, text, confidence)
                    result = self.ocr_engine.readtext(img_array, paragraph=False)
                    
                    if result:
                        page_text = " ".join([line[1] for line in result])
                        full_text += page_text + "\n"
            finally:
                doc.close()

            logger.info(f"Successfully extracted text from {pdf_path} (length: {len(full_text)})")

        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            if isinstance(e, (FileNotFoundError, RuntimeError)):
                 raise PDFOpenError(f"Could not open PDF {pdf_path}: {e}")
            raise OCRProcessingError(f"Failed to process PDF {pdf_path}: {e}")

        return full_text

    def get_word_bounding_boxes(self, pdf_path: Optional[str] = None) -> List[Dict]:
        """Get bounding boxes for all words in the PDF."""
        pdf_path = pdf_path or self.pdf_path
        if not pdf_path:
            raise ValueError("pdf_path must be provided")

        logger.info(f"Getting word bounding boxes for PDF: {pdf_path}")
        results = []

        try:
            doc = pymupdf.open(pdf_path)
            try:
                # FIX: Ensure we process all pages
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    # PDF page size in points (72 DPI)
                    pdf_width = page.rect.width
                    pdf_height = page.rect.height
                    
                    # Render at 150 DPI for OCR
                    dpi = 150
                    zoom = dpi / 72
                    pix = page.get_pixmap(dpi=dpi)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    img_array = np.array(img)
                    
                    # OCR returns coordinates in image space (150 DPI)
                    result = self.ocr_engine.readtext(img_array)

                    if result:
                        for line in result:
                            bbox = line[0]
                            full_line_text = line[1].strip()
                            if not full_line_text:
                                continue

                            # EasyOCR bbox points: top-left, top-right, bottom-right, bottom-left
                            # Convert from image space (150 DPI) to PDF space (72 DPI)
                            x0, y0 = bbox[0][0] / zoom, bbox[0][1] / zoom
                            x1, y1 = bbox[1][0] / zoom, bbox[1][1] / zoom
                            x2, y2 = bbox[2][0] / zoom, bbox[2][1] / zoom
                            
                            words = full_line_text.split()
                            if len(words) > 1:
                                # Estimate bounding boxes for individual words
                                line_width = x1 - x0
                                avg_char_width = line_width / len(full_line_text)
                                
                                current_x = x0
                                for word in words:
                                    word_len = len(word)
                                    word_width = word_len * avg_char_width
                                    results.append({
                                        "word": word,
                                        "page": page_num,
                                        "x": float(current_x),
                                        "y": float(y0),
                                        "width": float(word_width),
                                        "height": float(y2 - y0),
                                    })
                                    current_x += (word_len + 1) * avg_char_width # +1 for space
                            else:
                                results.append({
                                    "word": full_line_text,
                                    "page": page_num,
                                    "x": float(x0),
                                    "y": float(y0),
                                    "width": float(x1 - x0),
                                    "height": float(y2 - y0),
                                })
            finally:
                doc.close()

            logger.info(f"Found {len(results)} word boxes in {pdf_path}")

        except Exception as e:
            logger.error(f"Error getting bounding boxes for {pdf_path}: {e}")
            if isinstance(e, (FileNotFoundError, RuntimeError)):
                 raise PDFOpenError(f"Could not open PDF {pdf_path}: {e}")
            raise OCRProcessingError(f"Failed to process PDF {pdf_path}: {e}")

        return results

    def find_name_bounding_boxes(self, pdf_path: Optional[str] = None, text: Optional[str] = None) -> List[Dict]:
        """Match extracted names to their bounding boxes in the PDF."""
        pdf_path = pdf_path or self.pdf_path
        if not pdf_path:
            raise ValueError("pdf_path must be provided")

        logger.info(f"Finding name bounding boxes for PDF: {pdf_path}")

        if text is None:
            text = self.extract_text_from_pdf(pdf_path)

        ner_service = self.ner_service_class(text)
        names = ner_service.extract_names()
        word_boxes = self.get_word_bounding_boxes(pdf_path)

        name_boxes = []

        try:
            for name in names:
                name_parts = name.split()
                matched_boxes = []

                for part in name_parts:
                    for word_box in word_boxes:
                        if word_box["word"] == part:
                            matched_boxes.append(word_box)
                            break

                if matched_boxes:
                    min_x = min(b["x"] for b in matched_boxes)
                    min_y = min(b["y"] for b in matched_boxes)
                    max_x = max(b["x"] + b["width"] for b in matched_boxes)
                    max_y = max(b["y"] + b["height"] for b in matched_boxes)

                    name_boxes.append({
                        "name": name,
                        "page": matched_boxes[0]["page"],
                        "x": min_x,
                        "y": min_y,
                        "width": max_x - min_x,
                        "height": max_y - min_y,
                    })

            logger.info(f"Found {len(name_boxes)} name bounding boxes")

        except Exception as e:
            logger.error(f"Error matching names to bounding boxes: {e}")
            raise OCRProcessingError(f"Failed to match names to boxes: {e}")

        return name_boxes
