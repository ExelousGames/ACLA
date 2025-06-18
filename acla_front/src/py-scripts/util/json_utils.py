import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, date
from enum import Enum
from typing import Any
from uuid import UUID

class DataclassJSONEncoder(json.JSONEncoder):
    """
    JSON encoder that handles dataclasses, Enums, dates, and other common types.
    """
    def default(self, obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, Enum):
            return obj.value  # or obj.name if you prefer enum names
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)

class DataclassJSONUtility:
    """
    Utility class for converting between dataclasses and JSON.
    """
    @staticmethod
    def to_json(obj: Any, indent: int = None, **kwargs) -> str:
        """
        Convert a dataclass (or any supported object) to JSON string.
        
        Args:
            obj: The object to serialize
            indent: Pretty-printing indent level
            **kwargs: Additional arguments for json.dumps()
            
        Returns:
            JSON string representation of the object
        """
        return json.dumps(obj, cls=DataclassJSONEncoder, indent=indent, **kwargs)
    
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