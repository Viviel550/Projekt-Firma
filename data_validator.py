"""
Data validation and quality assessment
"""

import re
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class DataValidator:
    """Handles data validation and quality assessment"""
    
    def __init__(self):
        self.required_fields = [
            'customer_name', 'po_number', 'material_code', 
            'shipping_street', 'shipping_postcode'
        ]
        
        self.validation_patterns = {
            'postcode': r'\d{4}\s*[A-Z]{2}',
            'ppg_code': r'PPG\d+',
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        }
    
    def evaluate_quality(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate extraction quality"""
        quality_metrics = {
            'completeness': 0.0,
            'confidence': 0.0,
            'required_fields_present': False,
            'warnings': []
        }
        
        # Check completeness
        present_fields = sum(1 for field in self.required_fields 
                           if field in extracted_data and extracted_data[field])
        quality_metrics['completeness'] = present_fields / len(self.required_fields)
        
        # Check if all required fields are present
        quality_metrics['required_fields_present'] = all(
            field in extracted_data and extracted_data[field] 
            for field in self.required_fields
        )
        
        # Average confidence
        confidences = []
        if 'confidence' in extracted_data:
            confidences.append(float(extracted_data['confidence']))
        if 'classification_confidence' in extracted_data:
            confidences.append(float(extracted_data['classification_confidence']))
            
        if confidences:
            quality_metrics['confidence'] = sum(confidences) / len(confidences)
        
        # Generate warnings
        for field in self.required_fields:
            if field not in extracted_data or not extracted_data[field]:
                quality_metrics['warnings'].append(f"Missing required field: {field}")
        
        # Validate data formats
        if 'shipping_postcode' in extracted_data:
            postcode = extracted_data['shipping_postcode']
            if not re.match(self.validation_patterns['postcode'], postcode):
                quality_metrics['warnings'].append(f"Invalid postcode format: {postcode}")
        
        return quality_metrics
    
    def validate_extracted_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive data validation"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'missing_required_fields': []
        }
        
        # Check required fields
        for field in self.required_fields:
            if not data.get(field):
                validation_result['missing_required_fields'].append(field)
                validation_result['errors'].append(f"Missing required field: {field}")
        
        # Format validation
        if data.get('shipping_postcode'):
            if not self._validate_postcode(data['shipping_postcode']):
                validation_result['errors'].append("Invalid postcode format")
        
        if data.get('material_code'):
            if not self._validate_ppg_code(data['material_code']):
                validation_result['warnings'].append("Material code doesn't match PPG pattern")
        
        # Content validation
        if data.get('material_description'):
            required_keywords = ['sigma', 'ppg', 'rapid', 'gloss', 'lakverf', 'grondverf']
            has_keywords = any(keyword in data['material_description'].lower() 
                             for keyword in required_keywords)
            if not has_keywords:
                validation_result['warnings'].append("No material keywords found in description")
        
        if validation_result['errors']:
            validation_result['is_valid'] = False
        
        return validation_result
    
    def _validate_postcode(self, postcode: str) -> bool:
        """Validate Dutch postcode format"""
        return bool(re.match(self.validation_patterns['postcode'], postcode.strip()))
    
    def _validate_ppg_code(self, code: str) -> bool:
        """Validate PPG code format"""
        return bool(re.match(self.validation_patterns['ppg_code'], code.strip()))