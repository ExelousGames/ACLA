from pyaccsharedmemory import accSharedMemory
import sched, time
import csv
from typing import List, Any, Dict


asm = accSharedMemory()
recordedData = []
class ACCRecording:
    def __init__(self):
        return

    def startRecording(self):
        my_scheduler = sched.scheduler(time.time, time.sleep)
        my_scheduler.enter(0.01, 1, self.recordOnce, (my_scheduler,))
        my_scheduler.run()
        asm.close()

    def recordOnce(self,scheduler): 
        sm = asm.read_shared_memory()
        if  (sm is not None):
            # schedule the next call first
            scheduler.enter(1, 1, self.recordOnce, (scheduler,))
            recordedData.append(sm)
        else:
            self.objects_to_csv(recordedData,"acc_maps.csv")

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
                    flattened[full_key] = value
                    
            return flattened

    def objects_to_csv(self, objects: List[Any], filename: str, write_header: bool = True) -> None:
            """
            Convert a list of objects with nested structures to CSV
            """
            print(filename)
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
            
            # Convert to list and sort for consistent order
            fieldnames = sorted(all_fieldnames)
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if write_header:
                    writer.writeheader()
                
                for flattened in flattened_objects:
                    writer.writerow(flattened)



if __name__ == "__main__":
    recorder = ACCRecording()
    recorder.startRecording()