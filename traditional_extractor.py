import re
import pickle
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class TraditionalExtractor:
    """Handles traditional extraction methods using regex and simple ML"""
    
    def __init__(self):
        # Original PDF patterns
        self.patterns = {
            'customer_name': r'(?:Klant|Customer|Naam|voor):\s*([^\n\r]+)|(?:Hierbij de bestelling voor)\s*([^\n\r\.]+)',
            'po_number': r'(?:Boeknummer|PO|Order|BS|Orderreferentie)[\s:]*([A-Z0-9\-]+)',
            'material_code': r'(PPG\d+)',
            'material_description': r'(?:Sigma.*?(?:\d+\.?\d*\s*Ltr)|Opis.*?(?:\d+\.?\d*\s*Ltr)|Kit\s+\w+|Lakverf\s+\d+|Grondverf\s+\d+)',
            'shipping_street': r'(?:Adres|Address|Straat|Afleveradres):\s*([^\n\r,]+)',
            'shipping_postcode': r'(\d{4}\s*[A-Z]{2})',
            'colour_code': r'(?:Ral\s*\d+|No\.\d+\.?\d*)',
            'fan_code': r'(?:Fan|Waaier):\s*([^\n\r]+)',
            'shipping_condition': r'(?:Verzending|Levering|Shipping|Ophaaldatum):\s*([^\n\r]+)',
            'project_number': r'(?:Projectdossier).*?\n.*?(\d{8,})|(?:Projectdossier)\s*\n\s*(\d{8,})',
            'date': r'(?:Datum|Ophaaldatum):\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            'quantity': r'(\d+)\s*(?:stuks?|stuk|pieces?|pcs)\b',
            'reference_number': r'(?:Ref|Referentie|Referentienummer|Orderreferentie):\s*([^\n\r]+)'
        }
        
        # Email-specific patterns (looking for same fields as self.patterns)
        self.email_patterns = {
            'customer_name': r'(?:Hierbij de bestelling voor)\s*([^\n\r\.]+)|(?:bestelling voor)\s*([^\n\r\.]+)|(?:Klant|Customer):\s*([^\n\r]+)',
            'po_number': r'(?:Boeknummer|PO|Order|BS)[\s:]*([A-Z0-9\-]+)',
            'material_code': r'(PPG\d+)',
            'material_description': r'(?:Sigma\s+[^\n\r\t]+)|(?:Progold\s+[^\n\r\t]+)|(?:Kit\s+\w+)|(?:Lakverf\s+\d+)|(?:Grondverf\s+\d+)',
            'shipping_street': r'(?:Adres|Address|Straat|Afleveradres):\s*([^\n\r,]+)',
            'shipping_postcode': r'(\d{4}\s*[A-Z]{2})',
            'colour_code': r'(?:Kleur|Color)[\s\t]*(\d+)|(?:Ral\s*\d+|No\.\d+\.?\d*)',
            'fan_code': r'(?:Fan|Waaier):\s*([^\n\r]+)',
            'shipping_condition': r'(?:Verzending|Levering|Shipping|Ophaaldatum):\s*([^\n\r]+)',
            'project_number': r'(?:Subject:.*?)(\d{7,})',  # Extract from email subject
            'date': r'(?:Ophaaldatum|Pickup|Date):\s*(\d{1,2}[-/\.]\w{3,4}[-/\.]\d{2,4})|(?:Datum):\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            'quantity': r'(?:Aantal|Quantity)[\s\t]*(\d+)|(\d+)\s*(?:stuks?|stuk|pieces?|pcs)\b',
            'reference_number': r'(?:Orderreferentie|Order reference):\s*([A-Z0-9\-]+)|(?:Ref|Referentie):\s*([^\n\r]+)'
        }
        
        self.ml_model = None
        self.vectorizer = None
        self._load_model()
    
    def _load_model(self):
        """Load existing ML model or prepare for training"""
        try:
            with open('email_classifier.pkl', 'rb') as f:
                self.ml_model = pickle.load(f)
            with open('vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            logger.info("Loaded existing ML model")
        except FileNotFoundError:
            logger.info("No existing ML model found")
    
    def train_model(self):
        """Train simple ML model for classification"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.naive_bayes import MultinomialNB
            
            # Training data
            training_data = [
                ("bestelling order PPG materiaal", "order"),
                ("aanvraag offerte prijsopgave", "quote"),
                ("levering verzending adres", "delivery"),
                ("factuur betaling rekening", "invoice"),
                ("klacht probleem kwaliteit", "complaint"),
                ("Sigma Rapid Gloss bestelling", "order"),
                ("Kom ik ophalen", "pickup"),
                ("Afleveradres wijziging", "delivery"),
                ("BESTELLING", "order"),
                ("Boeknummer BS", "order"),
                ("Projectdossier", "order"),
                ("PPG Coatings", "order")
            ]
            
            texts, labels = zip(*training_data)
            
            self.vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
            X = self.vectorizer.fit_transform(texts)
            
            self.ml_model = MultinomialNB()
            self.ml_model.fit(X, labels)
            
            # Save model
            with open('email_classifier.pkl', 'wb') as f:
                pickle.dump(self.ml_model, f)
            with open('vectorizer.pkl', 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            logger.info("ML model trained and saved")
        except ImportError:
            logger.warning("sklearn not available, ML model training skipped")
    
    def extract_with_regex(self, text: str) -> Dict[str, Any]:
        """Extract data using regex patterns - handles both PDF and email formats"""
        extracted_data = {}
        
        # Determine if this is an email or PDF
        is_email = ('Subject:' in text or 'From:' in text or 'Hierbij de bestelling voor' in text)
        
        # Choose appropriate patterns
        patterns_to_use = self.email_patterns if is_email else self.patterns
        
        for field, pattern in patterns_to_use.items():
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                if field == 'project_number':
                    # Handle multiple groups in project_number regex
                    for match in matches:
                        if isinstance(match, tuple):
                            # Find the first non-empty group
                            project_num = next((group for group in match if group), None)
                            if project_num:
                                extracted_data[field] = project_num.strip()
                                break
                        else:
                            extracted_data[field] = match.strip()
                            break
                elif field == 'customer_name':
                    # Handle tuple matches for customer name (multiple groups)
                    match = matches[0]
                    if isinstance(match, tuple):
                        match_val = next((group for group in match if group), '')
                    else:
                        match_val = match
                    if match_val:
                        extracted_data[field] = match_val.strip()
                elif field == 'reference_number':
                    # Handle tuple matches for reference number
                    match = matches[0]
                    if isinstance(match, tuple):
                        match_val = next((group for group in match if group), '')
                    else:
                        match_val = match
                    if match_val:
                        extracted_data[field] = match_val.strip()
                elif field == 'date':
                    # Handle tuple matches for date (multiple date formats)
                    match = matches[0]
                    if isinstance(match, tuple):
                        match_val = next((group for group in match if group), '')
                    else:
                        match_val = match
                    if match_val:
                        extracted_data[field] = match_val.strip()
                elif field == 'quantity':
                    # Handle multiple quantities like material_code and colour_code
                    all_matches = []
                    for match in matches:
                        if isinstance(match, tuple):
                            match_val = next((group for group in match if group), '')
                        else:
                            match_val = match
                        if match_val:
                            all_matches.append(match_val.strip())
                    
                    if all_matches:
                        extracted_data[field] = all_matches[0]
                        if len(all_matches) > 1:
                            extracted_data[f'{field}s'] = all_matches
                elif field == 'material_description':
                    descriptions = []
                    for match in matches:
                        if isinstance(match, tuple):
                            match = match[0] if match[0] else (match[1] if len(match) > 1 else '')
                        parts = [part.strip() for part in str(match).split(';') if part.strip()]
                        descriptions.extend(parts)
                    
                    if descriptions:
                        extracted_data[field] = "; ".join(descriptions)
                        extracted_data['material_descriptions_list'] = descriptions
                elif field in ['material_code', 'colour_code']:
                    # Handle multiple matches
                    all_matches = []
                    for match in matches:
                        if isinstance(match, tuple):
                            match_val = next((group for group in match if group), '')
                        else:
                            match_val = match
                        if match_val:
                            all_matches.append(match_val.strip())
                    
                    if all_matches:
                        extracted_data[field] = all_matches[0]
                        if len(all_matches) > 1:
                            extracted_data[f'{field}s'] = all_matches
                else:
                    # Handle single match (could be tuple from multiple groups)
                    match = matches[0]
                    if isinstance(match, tuple):
                        match_val = next((group for group in match if group), '')
                    else:
                        match_val = match
                    
                    if match_val:
                        extracted_data[field] = match_val.strip()
        
        # Email-specific post-processing
        if is_email:
            extracted_data = self._enhance_email_extraction(extracted_data, text)
        
        return extracted_data
    
    def _enhance_email_extraction(self, extracted_data: Dict[str, Any], text: str) -> Dict[str, Any]:
        """Enhance email extraction with additional processing"""
        
        # Extract order items from email table
        order_items = self._extract_email_order_items(text)
        if order_items:
            extracted_data['order_items'] = order_items
            
            # Aggregate data from order items
            all_materials = []
            all_quantities = []
            all_colors = []
            
            for item in order_items:
                if item.get('material_name'):
                    all_materials.append(item['material_name'])
                if item.get('paint_type'):
                    all_materials.append(item['paint_type'])
                if item.get('quantity'):
                    all_quantities.append(item['quantity'])
                if item.get('color'):
                    all_colors.append(item['color'])
            
            # Update extracted data with aggregated info
            if all_materials and not extracted_data.get('material_description'):
                extracted_data['material_description'] = "; ".join(all_materials)
                extracted_data['material_descriptions_list'] = all_materials
            
            # Handle quantities - merge with existing quantities
            if all_quantities:
                existing_quantities = []
                if extracted_data.get('quantities'):
                    existing_quantities = extracted_data['quantities']
                elif extracted_data.get('quantity'):
                    existing_quantities = [extracted_data['quantity']]
                
                # Combine and deduplicate quantities
                combined_quantities = existing_quantities + all_quantities
                unique_quantities = []
                for qty in combined_quantities:
                    if qty not in unique_quantities:
                        unique_quantities.append(qty)
                
                extracted_data['quantity'] = unique_quantities[0] if unique_quantities else ''
                if len(unique_quantities) > 1:
                    extracted_data['quantities'] = unique_quantities
            
            if all_colors and not extracted_data.get('colour_code'):
                extracted_data['colour_code'] = all_colors[0]
                if len(all_colors) > 1:
                    extracted_data['colour_codes'] = all_colors
        
        return extracted_data
    
    def _extract_email_order_items(self, text: str) -> List[Dict[str, Any]]:
        """Extract order items from email table format"""
        items = []
        lines = text.split('\n')
        in_table = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect table start
            if 'EAN code' in line and 'Artikelnaam' in line:
                in_table = True
                continue
            
            # Skip headers and end markers
            if not in_table or 'Details van de bestelling' in line or 'Met vriendelijke groeten' in line:
                continue
            
            # Check if line contains table data
            if '\t' in line or (
                any(keyword in line for keyword in ['Kit', 'Lakverf', 'Grondverf', 'Sigma', 'Progold']) and
                any(char.isdigit() for char in line)
            ):
                # Split by tabs or multiple spaces
                parts = re.split(r'\t+|\s{3,}', line)
                
                if len(parts) >= 4:  # Ensure we have enough parts
                    item = {
                        'ean_code': parts[0] if parts[0] != 'Onbekend' else '',
                        'material_name': parts[1] if len(parts) > 1 else '',
                        'volume': parts[2] if len(parts) > 2 else '',
                        'color': parts[3] if len(parts) > 3 else '',
                        'paint_type': parts[4] if len(parts) > 4 else '',
                        'quantity': parts[5] if len(parts) > 5 else '1',
                        'customer_email': parts[6] if len(parts) > 6 else ''
                    }
                    items.append(item)
            
            # Stop at email signature
            if 'automatisch verzonden' in line:
                break
        
        return items
    
    def classify_document(self, text: str) -> Tuple[str, float]:
        """Classify document using traditional methods"""
        try:
            text_lower = text.lower()
            
            # Simple keyword-based classification with higher priority
            if 'bestelling' in text_lower or 'order' in text_lower or 'boeknummer' in text_lower:
                return 'order', 0.9
            elif 'factuur' in text_lower or 'invoice' in text_lower:
                return 'invoice', 0.9
            elif 'levering' in text_lower or 'delivery' in text_lower:
                return 'delivery', 0.9
            elif 'offerte' in text_lower or 'quote' in text_lower:
                return 'quote', 0.9
            
            # Use ML model as fallback
            if self.ml_model and self.vectorizer:
                X = self.vectorizer.transform([text_lower])
                prediction = self.ml_model.predict(X)[0]
                confidence = max(self.ml_model.predict_proba(X)[0])
                return prediction, confidence
            
            return "unknown", 0.5
        except Exception as e:
            logger.error(f"Error in traditional classification: {str(e)}")
            return "unknown", 0.5
    
    def extract_order_items(self, text: str) -> List[Dict[str, Any]]:
        """Extract order items from text with better quantity matching"""
        items = []
        lines = text.split('\n')
        
        # Enhanced quantity patterns
        quantity_patterns = [
            r'(\d+)\s+(?:stuks?|stuk|pieces?|pcs)\b',
            r'(?:stuks?|stuk|pieces?|pcs)[\s:]*(\d+)',
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for material lines
            if any(keyword in line.upper() for keyword in ['PPG', 'SIGMA', 'RAPID', 'GLOSS', 'PRIMER', 'ALLURE', 'KIT', 'LAKVERF', 'GRONDVERF']):
                item = {
                    'description': line,
                    'quantity': '',
                    'material_code': '',
                    'color': ''
                }
                
                # Extract material code
                ppg_match = re.search(r'(PPG\d+)', line)
                if ppg_match:
                    item['material_code'] = ppg_match.group(1)
                
                # Extract quantity from the same line or nearby lines
                quantity_found = False
                for pattern in quantity_patterns:
                    quantity_match = re.search(pattern, line, re.IGNORECASE)
                    if quantity_match:
                        qty = quantity_match.group(1)
                        try:
                            qty_int = int(qty)
                            if 1 <= qty_int <= 10000:  # Reasonable quantity range
                                item['quantity'] = qty
                                quantity_found = True
                                break
                        except ValueError:
                            continue
                
                # Extract color
                color_match = re.search(r'(RAL\s*\d+|No\.\s*\d+)', line, re.IGNORECASE)
                if color_match:
                    item['color'] = color_match.group(1)
                
                items.append(item)
        
        return items