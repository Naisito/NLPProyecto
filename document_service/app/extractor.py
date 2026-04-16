import io
import pdfplumber
import docx
from docx.text.paragraph import Paragraph
from docx.table import Table
import pytesseract
from PIL import Image

def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(x_tolerance=2, y_tolerance=3)
            if page_text:
                lines = [line.strip() for line in page_text.split('\n') if line.strip()]
                text_parts.append("\n".join(lines))
    return "\n\n".join(text_parts)
    
def extract_text_from_txt_bytes(file_bytes: bytes, encoding: str = "utf-8") -> str:
    return file_bytes.decode(encoding, errors="replace")

def extract_text_from_docx_bytes(file_bytes: bytes) -> str:
    doc = docx.Document(io.BytesIO(file_bytes))
    full_text = []
    
    for element in doc.element.body:
        if element.tag.endswith('p'):
            para = Paragraph(element, doc)
            if para.text.strip():
                full_text.append(para.text)
        elif element.tag.endswith('tbl'):
            table = Table(element, doc)
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells]
                full_text.append(" | ".join(row_cells))
                
    return "\n".join(full_text)

def extract_text_from_image_bytes(file_bytes: bytes, lang: str = "spa") -> str:
    image = Image.open(io.BytesIO(file_bytes)).convert("L")
    return pytesseract.image_to_string(image, lang=lang).strip()
