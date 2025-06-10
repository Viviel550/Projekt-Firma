import logging
import os
import re
from typing import List, Dict, Any, Tuple
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    pipeline
)

logger = logging.getLogger(__name__)

class TransformersHandler:
    """Handles transformer-based extraction methods"""
    
    def __init__(self):
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize transformer models"""
        try:
            logger.info("Initializing transformer models...")
            
            # Document classification model
            self.tokenizer_classifier = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.classifier_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
            
            # Named Entity Recognition model
            self.tokenizer_ner = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
            self.ner_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
            
            # Create pipelines
            self.ner_pipeline = pipeline("ner", model=self.ner_model, tokenizer=self.tokenizer_ner)
            self.classifier_pipeline = pipeline("text-classification", model=self.classifier_model, tokenizer=self.tokenizer_classifier)
            
            logger.info("Transformer models loaded successfully")
            
            # Optional fine-tuning
            self._load_fine_tuned_models()
            
        except Exception as e:
            logger.error(f"Error initializing transformer models: {str(e)}")
            raise
    
    def _load_fine_tuned_models(self):
        """Load fine-tuned models if available"""
        model_path = 'fine_tuned_ner_model'
        
        if os.path.exists(model_path):
            logger.info("Loading fine-tuned NER model...")
            self.ner_model = AutoModelForTokenClassification.from_pretrained(model_path)
            self.ner_pipeline = pipeline("ner", model=self.ner_model, tokenizer=self.tokenizer_ner)
        else:
            logger.info("No fine-tuned model found. Using base model.")
    
    def extract_structured_info(self, text: str) -> Dict[str, Any]:
        """Extract structured information using NER"""
        try:
            # Split text into chunks for processing
            max_length = 512
            chunks = self._split_text_into_chunks(text, max_length)
            
            all_entities = []
            for chunk in chunks:
                chunk_entities = self.ner_pipeline(chunk)
                all_entities.extend(chunk_entities)
            
            # Process NER results
            structured_data = self._process_ner_results(all_entities, text)
            
            # Enhance with context
            structured_data = self._enhance_with_context(structured_data, text)
            
            return structured_data
        except Exception as e:
            logger.error(f"Error in structured extraction: {str(e)}")
            return {}
    
    def classify_document(self, text: str) -> Tuple[str, float]:
        """Classify document using transformer model"""
        try:
            # Truncate text if too long
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            result = self.classifier_pipeline(text)[0]
            label = result['label']
            score = result['score']
            
            # Map model labels to our categories
            label_mapping = {
                'LABEL_0': 'order',
                'LABEL_1': 'invoice',
                'LABEL_2': 'delivery',
                'LABEL_3': 'quote',
                'LABEL_4': 'complaint'
            }
            
            return label_mapping.get(label, label), score
        except Exception as e:
            logger.error(f"Error in transformer classification: {str(e)}")
            return "unknown", 0.5
    
    def _split_text_into_chunks(self, text: str, max_length: int) -> List[str]:
        """Split text into smaller chunks for transformer processing"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_token_length = len(word.split()) + 1
            
            if current_length + word_token_length > max_length:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_token_length
            else:
                current_chunk.append(word)
                current_length += word_token_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _process_ner_results(self, entities: List[Dict[str, Any]], original_text: str) -> Dict[str, Any]:
        """Process NER results into structured data"""
        # Group entities by type
        grouped_entities = {}
        for entity in entities:
            entity_type = entity.get('entity')
            if entity_type.startswith('B-'):
                entity_type = entity_type[2:]
                
            if entity_type not in grouped_entities:
                grouped_entities[entity_type] = []
                
            grouped_entities[entity_type].append({
                'word': entity.get('word'),
                'score': entity.get('score'),
                'start': entity.get('start'),
                'end': entity.get('end')
            })
        
        # Map entity types to document fields
        entity_mapping = {
            'ORG': 'customer_name',
            'PER': 'contact_person',
            'LOC': 'shipping_location',
            'MISC': 'miscellaneous',
        }
        
        structured_data = {}
        for entity_type, entities_list in grouped_entities.items():
            if entity_type in entity_mapping:
                field_name = entity_mapping[entity_type]
                best_entity = max(entities_list, key=lambda x: x['score'])
                structured_data[field_name] = best_entity['word']
        
        # Extract custom entities
        structured_data = self._extract_custom_entities(structured_data, original_text)
        
        return structured_data
    
    def _extract_custom_entities(self, data: Dict[str, Any], text: str) -> Dict[str, Any]:
        """Extract domain-specific entities using rules"""
        import re
        
        # Extract PO numbers
        po_matches = re.findall(r'(?:BS|Boeknummer)[\s:]*(\d+)', text, re.IGNORECASE)
        if po_matches:
            data['po_number'] = po_matches[0]
        
        # Extract PPG material codes
        ppg_matches = re.findall(r'PPG\d+', text)
        if ppg_matches:
            data['material_codes'] = ppg_matches
            if not data.get('material_code') and ppg_matches:
                data['material_code'] = ppg_matches[0]
        
        # Extract postal codes
        postcode_matches = re.findall(r'\b\d{4}\s*[A-Z]{2}\b', text)
        if postcode_matches:
            data['shipping_postcode'] = postcode_matches[0]
        
        # Extract dates
        date_matches = re.findall(r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b', text)
        if date_matches:
            data['date'] = date_matches[0]
        
        # Updated project number extraction - handles both formats
        project_matches = re.findall(r'(?:Projectdossier).*?\n.*?(\d{8,})', text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if not project_matches:
            # Alternative pattern for direct format
            project_matches = re.findall(r'(?:Projectdossier)\s*\n\s*(\d{8,})', text, re.IGNORECASE | re.MULTILINE)
        if project_matches:
            data['project_number'] = project_matches[0].strip()
        
        return data
    
    def _enhance_with_context(self, data: Dict[str, Any], text: str) -> Dict[str, Any]:
        """Add contextual information based on text analysis"""
        # Document type determination
        if "BESTELLING" in text.upper():
            data['document_type'] = 'order'
        elif "FACTUUR" in text.upper():
            data['document_type'] = 'invoice'
        elif "LEVERING" in text.upper():
            data['document_type'] = 'delivery'
        
        return data