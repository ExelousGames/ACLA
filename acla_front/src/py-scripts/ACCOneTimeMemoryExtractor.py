from pyaccsharedmemory import accSharedMemory
import sched, time
import csv
from typing import List, Any, Dict
import os
from util.json_utils import DataclassJSONUtility
from util.clean_encode import cleanEncoding

recordedData = []

class ACCRecording:
    def __init__(self):
        self.asm = accSharedMemory()
        return

    def startRecording(self):
        self.recordOnce()    

    def recordOnce(self): 
        sm = self.asm.read_shared_memory()
        if  (sm is not None):
            # !!!!!! must keep this to communicate with frontend
            print(DataclassJSONUtility.to_json(sm, indent=2).rstrip())
        else:
            self.asm.close()
            # self.objects_to_csv(recordedData,"acc_maps.csv")
            return
            

    def flatten_object(self, obj: Any, prefix: str = '') -> Dict[str, Any]:
            """
            Recursively flatten an object's attributes, including nested objects
            """
            flattened = {}
            
            # Skip non-object attributes or None values
            if not hasattr(obj, '__dict__') or obj is None:
                return flattened
            
            for key, value in vars(obj).items():
                # Skip private attributes
                if key.startswith('_'):
                    continue
                    
                full_key = f"{prefix}{key}" if prefix else key
                
                # Handle nested objects recursively
                if hasattr(value, '__dict__'):
                    nested_flattened = self.flatten_object(value, prefix=f"{full_key}_")
                    flattened.update(nested_flattened)
                # Handle basic types
                else:
                    valueFixed = cleanEncoding(value)

                    if not self.is_blank(valueFixed):
                        flattened[full_key] = valueFixed
                    
            return flattened
        
    def is_blank(self,value):
        """Check if value is None, empty, or whitespace-only for any type"""
        if value is None:
            return True
        if isinstance(value, (str, bytes)):
            return not str(value).strip()
        if isinstance(value, (list, dict, set, tuple)):
            return not bool(value)
        return False
    

    
if __name__ == "__main__":
    recorder = ACCRecording()
    recorder.startRecording()