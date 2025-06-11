import logging
import os
from typing import Dict, Any, List, Tuple
from datetime import datetime

# Change these lines from relative to absolute imports
from transformers_handler import TransformersHandler
from traditional_extractor import TraditionalExtractor
from pdf_processor import PDFProcessor
from email_processor import EmailProcessor
from data_validator import DataValidator

logger = logging.getLogger(__name__)

class EnhancedDataExtractor:
    """Main extractor class that coordinates all extraction methods"""
    
    def __init__(self, use_transformers=True):
        self.use_transformers = use_transformers
        
        # Initialize handlers
        self.transformers_handler = None
        self.traditional_extractor = TraditionalExtractor()
        self.pdf_processor = PDFProcessor()
        self.email_processor = EmailProcessor()
        self.validator = DataValidator()
        
        if self.use_transformers:
            try:
                self.transformers_handler = TransformersHandler()
                logger.info("Transformers initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize transformers: {str(e)}")
                logger.warning("Falling back to traditional methods...")
                self.use_transformers = False
        
        if not self.use_transformers:
            self.traditional_extractor.train_model()
    
    def extract_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract data from PDF file"""
        try:
            # Extract text from PDF
            text = self.pdf_processor.extract_text(pdf_path)
            
            if not text.strip():
                logger.warning(f"No text extracted from PDF: {pdf_path}")
                return {}
            
            # Process the extracted text
            return self._process_text(text, source_type='pdf')
            
        except Exception as e:
            logger.error(f"Error extracting from PDF {pdf_path}: {str(e)}")
            return {}
    
    def extract_from_email(self, email_content: str) -> Dict[str, Any]:
        """Extract data from email content"""
        return self._process_text(email_content, source_type='email')
    
    def _process_text(self, text: str, source_type: str = 'unknown') -> Dict[str, Any]:
        """Process text using available extraction methods"""
        extracted_data = {}
        
        # Classify document
        doc_type, confidence = self._classify_document(text)
        extracted_data['document_type'] = doc_type
        extracted_data['classification_confidence'] = confidence
        
        # Extract using transformers if available
        if self.use_transformers and self.transformers_handler:
            transformer_data = self.transformers_handler.extract_structured_info(text)
            extracted_data.update(transformer_data)
        
        # Extract using traditional methods as fallback/supplement
        traditional_data = self.traditional_extractor.extract_with_regex(text)
        
        # Merge data - prefer transformer results
        for key, value in traditional_data.items():
            if key not in extracted_data or not extracted_data[key]:
                extracted_data[key] = value
        
        # Source-specific processing
        if source_type == 'email':
            extracted_data = self.email_processor.enhance_extraction(extracted_data, text)
        
        # Extract order items for orders
        if doc_type == 'order':
            order_items = self._extract_order_items(text)
            if order_items:
                extracted_data['order_items'] = order_items
                
                # Extract quantities from order items
                item_quantities = [item.get('quantity', '') for item in order_items if item.get('quantity')]
                if item_quantities:
                    # Update quantities in extracted data
                    if not extracted_data.get('quantities'):
                        extracted_data['quantities'] = item_quantities
                        extracted_data['quantity'] = item_quantities[0]
                    else:
                        # Merge with existing quantities
                        existing_quantities = extracted_data.get('quantities', [])
                        combined_quantities = existing_quantities + item_quantities
                        # Remove duplicates while preserving order
                        unique_quantities = []
                        for qty in combined_quantities:
                            if qty and qty not in unique_quantities:
                                unique_quantities.append(qty)
                        extracted_data['quantities'] = unique_quantities
                        extracted_data['quantity'] = unique_quantities[0] if unique_quantities else ''
        
        return extracted_data
    
    def _classify_document(self, text: str) -> Tuple[str, float]:
        """Classify document type"""
        if self.use_transformers and self.transformers_handler:
            return self.transformers_handler.classify_document(text)
        else:
            return self.traditional_extractor.classify_document(text)
    
    def _extract_order_items(self, text: str) -> List[Dict[str, Any]]:
        """Extract order items from text"""
        return self.traditional_extractor.extract_order_items(text)
    
    def read_emails_from_imap(self, server: str, username: str, password: str, 
                            folder: str = 'INBOX', unread_only: bool = True, 
                            mark_as_read: bool = True) -> List[Dict[str, Any]]:
        """Read emails from IMAP server"""
        return self.email_processor.read_from_imap(
            server, username, password, folder, self, unread_only, mark_as_read
        )
    def save_to_excel(self, data_list: List[Dict[str, Any]], output_file: str) -> bool:
        """Save extracted data to Excel file"""
        from excel_exporter import ExcelExporter
        exporter = ExcelExporter()
        return exporter.save_to_excel(data_list, output_file)
    
    def evaluate_extraction_quality(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate extraction quality"""
        return self.validator.evaluate_quality(extracted_data)
    
    def export_to_json(self, data_list: List[Dict[str, Any]], output_file: str) -> bool:
        """Export data to JSON file"""
        from json_exporter import JSONExporter
        exporter = JSONExporter()
        return exporter.export_to_json(data_list, output_file)
    
    def process_directory(self, directory_path: str, output_file: str) -> bool:
        """Process all PDF files in directory"""
        all_data = []
        
        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(directory_path, filename)
                extracted_data = self.extract_from_pdf(pdf_path)
                extracted_data['source_file'] = filename
                all_data.append(extracted_data)
        
        if all_data:
            return self.save_to_excel(all_data, output_file)
        else:
            logger.warning("No data found to process")
            return False

# Test if running directly
if __name__ == "__main__":
    print("Testing EnhancedDataExtractor...")
    try:
        extractor = EnhancedDataExtractor()
        print("✓ EnhancedDataExtractor initialized successfully")
    except Exception as e:
        print(f"✗ Error: {e}")