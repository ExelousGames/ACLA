from pyaccsharedmemory import accSharedMemory
import sched, time
import csv
from typing import List, Any, Dict
import os
from util.json_utils import DataclassJSONUtility
from util.clean_encode import cleanEncoding
import sys

recordedData = []

class ACCRecording:
    def __init__(self):
        self.asm = accSharedMemory()
        return

    def startRecording(self,full_path):
        sm = self.asm.read_shared_memory()
        if  (sm is not None):
            #record once to clean or create the file
            self.write_object_to_csv(sm,full_path)
            #start to record the session
            my_scheduler = sched.scheduler(time.time, time.sleep)
            my_scheduler.enter(0.1, 1, self.recordOnce, (my_scheduler,full_path))
            my_scheduler.run()
            
        else:
            self.asm.close()

    def recordOnce(self,scheduler,full_path): 
        sm = self.asm.read_shared_memory()
        if  (sm is not None):
            # schedule the next call first
            scheduler.enter(1, 1, self.recordOnce, (scheduler,full_path))

            self.append_object_to_csv(sm,full_path)

            # !!!!!! must keep this to communicate with frontend
            print(DataclassJSONUtility.to_json(sm, indent=2).rstrip())
        else:
            self.asm.close()

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
    def append_object_to_csv(self, object: Any, filename: str) -> None:
            """
            Convert object with nested structures to CSV
            """
            if not filename:
                raise ValueError("No filename provided")
            
            if not object:
                return
                
            # Get all possible fieldnames from all objects
            all_fieldnames = set()
            flattened_objects = []
            
   
            flattened = self.flatten_object(object)
            flattened_objects.append(flattened)
            all_fieldnames.update(flattened.keys())
            
            
            with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(
                    csvfile,
                    fieldnames=all_fieldnames,
                    quoting=csv.QUOTE_MINIMAL,  # This will handle special characters
                    escapechar='\\'  # Add escape character
                )
                
                
                for flattened in flattened_objects:
                    writer.writerow(flattened)
    
    def write_object_to_csv(self, objects: Any, filename: str, write_header: bool = True) -> None:
            """
            Convert a list of objects with nested structures to CSV
            """
            if not filename:
                raise ValueError("No filename provided")
            
            if not objects:
                return
                
            # Get all possible fieldnames from all objects
            all_fieldnames = set()
            flattened_objects = []
            
            flattened = self.flatten_object(objects)
            flattened_objects.append(flattened)
            all_fieldnames.update(flattened.keys())
            
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(
                    csvfile,
                    fieldnames=all_fieldnames,
                    quoting=csv.QUOTE_MINIMAL,  # This will handle special characters
                    escapechar='\\'  # Add escape character
                )
                
                if write_header:
                    writer.writeheader()
                
                for flattened in flattened_objects:
                    writer.writerow(flattened)
                   
    def write_objects_to_csv(self, objects: List[Any], filename: str, write_header: bool = True) -> None:
            """
            Convert a list of objects with nested structures to CSV
            """
            if not filename:
                raise ValueError("No filename provided")
            
            if not objects:
                return
                
            # Get all possible fieldnames from all objects
            all_fieldnames = set()
            flattened_objects = []
            
            for obj in objects:
                flattened = self.flatten_object(obj)
                flattened_objects.append(flattened)
                all_fieldnames.update(flattened.keys())
            
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(
                    csvfile,
                    fieldnames=all_fieldnames,
                    quoting=csv.QUOTE_MINIMAL,  # This will handle special characters
                    escapechar='\\'  # Add escape character
                )
                
                if write_header:
                    writer.writeheader()
                
                for flattened in flattened_objects:
                    writer.writerow(flattened)

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
    folder = sys.argv[1]
    filename = sys.argv[2]

    full_path = os.path.join(folder,filename)
    os.makedirs(folder,exist_ok=True)
    
    recorder.startRecording(full_path)