import threading
import json
import logging
import time
import queue
from typing import List, Dict, Any, Callable, Optional
import pandas as pd
from streamlit.runtime.scriptrunner import add_script_run_ctx

# Interface for progress updates
class BatchProgressObserver:
    def on_start(self, total_items: int): pass
    def on_progress(self, current_index: int, total_items: int, message: str = ""): pass
    def on_log(self, message: str): pass
    def on_complete(self, success_count: int, error_count: int, message: str = "Completed"): pass
    def on_error(self, error_message: str): pass

class StreamlitBatchObserver(BatchProgressObserver):
    """
    Observer implementation that updates Streamlit UI components.
    Safe for cross-thread updates if components are passed correctly.
    """
    def __init__(self, progress_bar, status_text_placeholder, log_area_placeholder):
        self.progress_bar = progress_bar
        self.status_text = status_text_placeholder
        self.log_area = log_area_placeholder
        self.logs = []
        self._lock = threading.Lock()

    def on_start(self, total_items: int):
        with self._lock:
            self.logs = []
            if self.progress_bar:
                try: self.progress_bar.progress(0.0)
                except: pass
            if self.status_text:
                try: self.status_text.text("Starting...")
                except: pass

    def on_progress(self, current_index: int, total_items: int, message: str = ""):
        with self._lock:
            if total_items > 0 and self.progress_bar:
                progress = min(1.0, float(current_index) / total_items)
                try: self.progress_bar.progress(progress)
                except: pass
            
            if self.status_text:
                status_msg = f"Processing ({current_index + 1}/{total_items}) - {message}"
                try: self.status_text.text(status_msg)
                except: pass

    def on_log(self, message: str):
        with self._lock:
            self.logs.append(message)
            if len(self.logs) > 1000:
                self.logs.pop(0)
            
            if self.log_area:
                try:
                    # Show more context (last 50 lines) so users can see previous segment results
                    display_text = "\n".join(self.logs[-50:])
                    self.log_area.code(display_text, language="text")
                except: pass
    
    def on_complete(self, success_count: int, error_count: int, message: str = "Completed"):
        with self._lock:
            if self.progress_bar:
                try: self.progress_bar.progress(1.0)
                except: pass
            if self.status_text:
                try: self.status_text.text(f"{message} (Success: {success_count}, Errors: {error_count})")
                except: pass

    def on_error(self, error_message: str):
         with self._lock:
            if self.status_text:
                try: self.status_text.text(f"Error: {error_message}")
                except: pass
            self.on_log(f"CRITICAL ERROR: {error_message}")

class BatchAnnotationService:
    def __init__(self, gemini_analyzer, observer: BatchProgressObserver):
        self.analyzer = gemini_analyzer
        self.observer = observer
        self._stop_event = threading.Event()
        self.is_running = False
        self.task_queue = queue.Queue(maxsize=5) # Buffer size for tasks

    def stop(self):
        self._stop_event.set()
        current_thread_name = threading.current_thread().name
        msg = f"🛑 Stop requested by user on thread '{current_thread_name}'. Waiting for worker thread..."
        print(msg) # Ensure it prints to console
        self.observer.on_log(msg)
        
        # Clear queue to unblock potential puts if stuck
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
                self.task_queue.task_done()
            except queue.Empty:
                break

    def _worker_process(self, result_callback, label_name_to_id, label_mapping, total_items):
        """Worker thread function to process analysis requests from the queue."""
        add_script_run_ctx(threading.current_thread())
        thread_name = threading.current_thread().name
        self.observer.on_log(f"Worker thread '{thread_name}' started.")
        
        success_count = 0
        error_count = 0
        processed_count = 0

        while self.is_running or not self.task_queue.empty():
            try:
                # Wait for a task with timeout to check for stop signal occasionally
                try:
                    task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    if not self.is_running:
                        break
                    continue
                
                if self._stop_event.is_set():
                    self.task_queue.task_done()
                    break

                idx, segment_df, track_config, current_labels_display, context_sub_labels = task
                
                # Update progress (approximate, based on dequeued items)
                # Note: The producer loop updates progress based on queuing, but maybe meaningful progress is analysis completion?
                # Let's update progress here based on completion.
                
                start_time = time.time()
                thread_info = f"Thread: {thread_name} (ID: {threading.get_ident()})"
                
                self.observer.on_log(f"    [Seg #{idx}] Calling Gemini Analyzer on {thread_info}...")
                
                # --- HEAVY LIFTING (Plotting + API) ---
                try:
                    result = self.analyzer.analyze_segment_json(
                        segment_df,
                        track_config=track_config,
                        current_labels=current_labels_display,
                        available_sub_labels_context=context_sub_labels
                    )
                    
                    self.observer.on_log(f"    [Seg #{idx}] Gemini Analyzer returned on {thread_info}.")
                    
                    elapsed = time.time() - start_time
                    metrics = result.get("metrics", {})
                    
                    if metrics:
                         self.observer.on_log(f"   [Seg #{idx}] API Footprint: Req ~{metrics.get('request_text_kb',0):.1f}KB txt + ~{metrics.get('request_imgs_kb',0):.1f}KB img. Resp: ~{metrics.get('response_kb',0):.1f}KB. Time: {metrics.get('duration_sec',elapsed):.2f}s")
                    else:
                         prompt_len = len(str(result.get("prompt", "")))
                         img_count = len(result.get("images", []))
                         self.observer.on_log(f"   [Seg #{idx}] Analysis API returned in {elapsed:.2f}s. Sent: {prompt_len} chars prompt, {img_count} images.")

                    if "error" in result:
                        self.observer.on_log(f"Segment #{idx} error: {result['error']}")
                        error_count += 1
                    else:
                        # Process JSON Response
                        suggestions, full_response_text = self._parse_response(result, idx)
                        
                        if suggestions or full_response_text:
                             log_msg = f"Segment #{idx}: Found {len(suggestions)} suggestions."
                             suggested_ids = []
                             for sugg in suggestions:
                                 lname = sugg.get("label")
                                 lid = label_name_to_id.get(lname)
                                 
                                 if not lid and lname in label_mapping:
                                     # The model returned a valid ID directly (e.g. MS1) instead of the mapped name
                                     # This can happen even if we prompt for names, due to context provided.
                                     lid = lname
                                     
                                 if lid:
                                     suggested_ids.append(lid)
                                     log_msg += f"\n  - {lname} -> {lid} ({sugg.get('confidence', 0):.2f}): {sugg.get('reasoning', '')}"
                                 else:
                                     # Log skipped but don't add to suggested_ids
                                     log_msg += f"\n  - (skipped) {lname} not found in map."
                             
                             self.observer.on_log(log_msg)
                             
                             # Execute Callback to persist/apply changes and save notes
                             self.observer.on_log(f"   [Seg #{idx}] Applying labels and saving notes via callback...")
                             result_callback(idx, suggested_ids, full_response_text)
                             success_count += 1
                        else:
                            self.observer.on_log(f"Segment #{idx}: No valid suggestions or response returned.")

                except Exception as e:
                    self.observer.on_log(f"Error executing analysis for segment #{idx}: {e}")
                    error_count += 1

                processed_count += 1
                self.observer.on_progress(processed_count, total_items, f"Analyzed segment #{idx}...")
                self.task_queue.task_done()
                
            except Exception as e:
                self.observer.on_error(f"Worker thread error: {e}")
                
        self.observer.on_complete(success_count, error_count)
        self.observer.on_log(f"Worker thread '{thread_name}' finished.")

    def run_batch(
        self,
        process_indices: List[int],
        annotations: List[Any], # List of annotation objects
        df: pd.DataFrame,
        track_config: Dict,
        label_mapping: Dict,
        label_name_to_id: Dict,
        main_guidelines: Dict,
        label_categories: Dict,
        result_callback: Callable[[int, List[str], str], None] # Callback to save results (idx, labels, notes)
    ):
        self._stop_event.clear()
        self.is_running = True
        total = len(process_indices)
        
        self.observer.on_start(total)
        current_thread_name = threading.current_thread().name
        self.observer.on_log(f"Starting batch producer for {total} segments in thread: {current_thread_name}")

        # Start Worker Thread
        worker_thread = threading.Thread(
            target=self._worker_process, 
            args=(result_callback, label_name_to_id, label_mapping, total),
            name="GeminiWorkerThread",
            daemon=True
        )
        add_script_run_ctx(worker_thread)
        worker_thread.start()

        try:
            for i, idx in enumerate(process_indices):
                if self._stop_event.is_set():
                    current_thread_name = threading.current_thread().name
                    self.observer.on_log(f"⚠️ Process stopped by user at segment index {idx} (processed {i}/{total} items) on thread '{current_thread_name}'. Terminating batch.")
                    break

                try:
                    # Validate Index
                    if idx < 0 or idx >= len(annotations):
                        self.observer.on_log(f"Invalid segment index: {idx}")
                        continue

                    ann = annotations[idx]
                    segment_df = df.iloc[int(ann.start_index):int(ann.end_index)]
                    
                    if segment_df.empty:
                        self.observer.on_log(f"Segment #{idx} is empty. Skipping.")
                        continue
                        
                    # Prepare Context
                    current_labels_display = [label_mapping.get(l, l) for l in ann.labels]
                    
                    context_sub_labels = self._build_sub_context(
                        current_labels_display, 
                        label_name_to_id, 
                        main_guidelines, 
                        label_categories,
                        label_mapping
                    )
                    
                    # Log queuing
                    # self.observer.on_log(f"--> [Seg #{idx}] Queuing task...")
                    
                    # Push to Queue (will block if full, effectively throttling the producer)
                    self.task_queue.put((idx, segment_df, track_config, current_labels_display, context_sub_labels))
                    
                except Exception as e:
                    self.observer.on_log(f"Error preparing segment #{idx}: {str(e)}")

            # Signal production is done, wait for queue to drain
            # Don't set is_running = False yet, wait for worker completion
            self.observer.on_log("Producer finished queuing tasks. Waiting for worker to complete...")
            self.task_queue.join()
            
        except Exception as e:
            self.observer.on_error(f"Fatal batch error: {str(e)}")
        finally:
            # Ensure worker stops if it hasn't already
            if worker_thread.is_alive():
                 worker_thread.join(timeout=5.0)
            
            # Now we are truly done
            self.is_running = False

    def _build_sub_context(self, current_labels, label_name_to_id, main_guidelines, label_categories, label_mapping):
        """Helper to build context strings for sub-labels."""
        context_list = []
        for lname in current_labels:
            lid = label_name_to_id.get(lname)
            if lid:
                # 1. Add context/guideline for the PARENT label itself if available
                if lid in main_guidelines:
                    desc = main_guidelines[lid]
                    context_list.append(f"Instruction for current label '{lname}': {desc}")
                
                # 2. Add available sub-labels
                if lid in label_categories:
                    sub_label_ids = label_categories[lid]
                    context_list.append(f"Available Sub-labels for {lname}:")
                    for sub_id in sub_label_ids:
                        full_name = label_mapping.get(sub_id, sub_id)
                        desc = main_guidelines.get(sub_id, "")
                        line = f"  - {full_name}"
                        if desc:
                            line += f": {desc}"
                        context_list.append(line)
        return context_list

    def _parse_response(self, result, idx):
        """Returns tuple of (suggestions_list, raw_json_text)"""
        raw_to_parse = result.get("response_json", "")
        # Keep original raw text for notes
        original_raw = raw_to_parse
        
        if not raw_to_parse: return [], ""
        
        # Cleanup markdown
        if "```json" in raw_to_parse:
            raw_to_parse = raw_to_parse.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_to_parse:
            raw_to_parse = raw_to_parse.split("```")[1].split("```")[0].strip()
        
        try:
            data = json.loads(raw_to_parse)
            summary = data.get("analysis_summary", "")
            if summary:
                self.observer.on_log(f"Segment #{idx} Summary: {summary}")
            return data.get("suggested_labels", []), original_raw
        except json.JSONDecodeError:
            self.observer.on_log(f"Segment #{idx}: Invalid JSON response.")
            return [], original_raw
