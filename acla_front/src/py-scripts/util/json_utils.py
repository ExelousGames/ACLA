import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, date
from enum import Enum
from typing import Any
from uuid import UUID
import base64
class DataclassJSONEncoder(json.JSONEncoder):
    """
    JSON encoder that handles dataclasses, Enums, dates, and other common types.
    """
    def default(self, obj: Any) -> Any:
        
        if is_dataclass(obj):
            return self._process_dataclass(obj)
        elif isinstance(obj, Enum):
            return obj.value  # or obj.name if you prefer enum names
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, bytes):
            return self._serialize_bytes(obj)
        return super().default(obj)
    
    def _process_dataclass(self, obj: Any) -> dict:
        """Convert dataclass to dict with custom string handling"""
        result = {}
        for field_name, field_value in asdict(obj).items():
            cleaned_value = self._process_value(field_value)
            if cleaned_value is not None:  # Skip None values if desired
                result[field_name] = cleaned_value
        return result
    
    def _process_value(self, value: Any) -> Any:
        """Recursively process values with special string handling"""
        if isinstance(value, (str)):
            return value.encode('ascii', errors='ignore').decode('ascii').replace('\u0000', '')
        elif isinstance(value, dict):
            return {k: self._process_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple, set)):
            return [self._process_value(v) for v in value]
        return value
    
    def _serialize_bytes(self, byte_data: bytes) -> str:
        """Handle bytes serialization with multiple options"""
        try:
            # First try UTF-8 decoding if it's text
            return byte_data.decode('utf-8')
        except UnicodeDecodeError:
            # Fall back to base64 for binary data
            return f"base64:{base64.b64encode(byte_data).decode('utf-8')}"
        
class DataclassJSONUtility:
    """
    Utility class for converting between dataclasses and JSON.
    """
    @staticmethod
    def to_json(obj: Any, compact: bool = True, **kwargs) -> str:
        """
        Convert object to JSON in compact format (no whitespace) by default
        
        Args:
            obj: Object to serialize
            compact: If True (default), removes all whitespace
            **kwargs: Additional arguments for json.dumps()
            
        Returns:
            JSON string without line breaks or indentation
        """
        encoder = DataclassJSONEncoder
        
        if compact:
            # Force these settings for compact output
            kwargs.update({
                'indent': None,
                'separators': (',', ':'),
                'sort_keys': False
            })
            
        return json.dumps(obj, cls=encoder, **kwargs)
    
    @staticmethod
    def to_dict(obj: Any) -> dict:
        """
        Convert a dataclass to dictionary, handling nested dataclasses and Enums.
        
        Args:
            obj: The dataclass instance to convert
            
        Returns:
            Dictionary representation of the dataclass
        """
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, Enum):
            return obj.value
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    @staticmethod
    def from_json(json_str: str, target_class: type, **kwargs) -> Any:
        """
        Convert JSON string back to a dataclass instance (basic implementation).
        Note: For complex cases, you'll need to implement custom deserialization.
        
        Args:
            json_str: JSON string to deserialize
            target_class: The dataclass type to convert to
            **kwargs: Additional arguments for json.loads()
            
        Returns:
            Instance of the target_class populated with data from JSON
        """
        data = json.loads(json_str, **kwargs)
        return target_class(**data)