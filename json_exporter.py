import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class JSONExporter:
    """Handles JSON export"""
    
    def export_to_json(self, data_list: List[Dict[str, Any]], output_file: str) -> bool:
        """Export data to JSON file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data_list, f, indent=4, ensure_ascii=False, default=str)
            logger.info(f"Data saved to JSON: {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving to JSON: {str(e)}")
            return False