import logging

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF text extraction"""
    
    def extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        
        # Try pdfplumber first (better for tables)
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except ImportError:
            logger.warning("pdfplumber not installed. Install with: pip install pdfplumber")
        except Exception as e:
            logger.warning(f"pdfplumber extraction error: {str(e)}")
        
        # Fallback to PyPDF2 if pdfplumber failed
        if not text.strip():
            try:
                import PyPDF2
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except ImportError:
                logger.error("PyPDF2 not installed. Install with: pip install PyPDF2")
            except Exception as e:
                logger.error(f"PyPDF2 extraction error: {str(e)}")
        
        return text