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


class BatchFileJobState:
    """
    Holds state for a Batch API file-based job to persist across Streamlit reruns.
    """
    def __init__(self):
        self.job_name: Optional[str] = None
        self.job_display_name: Optional[str] = None
        self.uploaded_file_name: Optional[str] = None  # File API uploaded file reference
        self.status: str = "idle"  # idle, preparing, uploading, submitted, polling, completed, failed, cancelled
        self.total_requests: int = 0
        self.segment_index_map: Dict[str, int] = {}  # Maps request key -> segment index
        self.error_message: Optional[str] = None
        self.results: List[Dict[str, Any]] = []
        self.poll_count: int = 0
        self.start_time: Optional[float] = None

class StreamlitBatchObserver(BatchProgressObserver):
    """
    Observer implementation that updates Streamlit UI components.
    Safe for cross-thread updates if components are passed correctly.
    """
    def __init__(self, progress_bar, status_text_placeholder, log_area_placeholder, logs: List[str] = None):
        self.progress_bar = progress_bar
        self.status_text = status_text_placeholder
        self.log_area = log_area_placeholder
        self.logs = logs if logs is not None else []
        self.current_progress = 0.0
        self.current_status = ""
        self._lock = threading.Lock()

    def restore_state(self, logs: Optional[List[str]], progress: float, status: str):
        """Restores the UI state from persisted service data."""
        with self._lock:
            # If logs provided and different from internal storage, update internal
            if logs is not None and logs is not self.logs:
                self.logs = logs
                
            self.current_progress = progress
            self.current_status = status
            
            if self.log_area and self.logs:
                try:
                    display_text = "\n".join(self.logs)
                    self.log_area.code(display_text, language="text", line_numbers=True)
                except Exception as e:
                    print(f"Error restoring log area: {e}")
            
            if self.progress_bar:
                try: self.progress_bar.progress(progress)
                except: pass
            
            if self.status_text and status:
                try: self.status_text.text(status)
                except: pass

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
            if total_items > 0:
                progress = min(1.0, float(current_index) / total_items)
                self.current_progress = progress
                if self.progress_bar:
                    try: self.progress_bar.progress(progress)
                    except: pass
            
            self.current_status = f"Processing ({current_index + 1}/{total_items}) - {message}"
            if self.status_text:
                try: self.status_text.text(self.current_status)
                except: pass

    def on_log(self, message: str):
        with self._lock:
            try:
                logging.info(f"[BatchLog]: {message}")
            except: pass

            self.logs.append(message)
            if len(self.logs) > 1000:
                self.logs.pop(0)
            
            if self.log_area:
                try:
                    # Show more context because text_area is scrollable
                    display_text = "\n".join(self.logs)
                    # Using code block instead of text_area avoids key conflicts and widget state issues during updates
                    # It also provides a copy button and looks more like a log
                    self.log_area.code(display_text, language="text", line_numbers=True)
                except Exception as e:
                    print(f"Error updating log area: {e}")
    
    def on_complete(self, success_count: int, error_count: int, message: str = "Completed"):
        with self._lock:
            self.current_progress = 1.0
            if self.progress_bar:
                try: self.progress_bar.progress(1.0)
                except: pass
            
            self.current_status = f"{message} (Success: {success_count}, Errors: {error_count})"
            if self.status_text:
                try: self.status_text.text(self.current_status)
                except: pass

    def on_error(self, error_message: str):
         with self._lock:
            self.current_status = f"Error: {error_message}"
            if self.status_text:
                try: self.status_text.text(self.current_status)
                except: pass
            self.on_log(f"CRITICAL ERROR: {error_message}")

class BatchAnnotationService:
    def __init__(self, gemini_analyzer, observer: BatchProgressObserver):
        self.analyzer = gemini_analyzer
        self.observer = observer
        self._stop_event = threading.Event()
        self.is_running = False
        self.task_queue = queue.Queue(maxsize=5) # Buffer size for tasks
        self.logs = []
        self.progress_value = 0.0
        self.status_message = ""

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

    def _worker_process(self, result_callback, label_name_to_id, label_mapping, total_items, context_padding=200, overlay_config=None):
        """Worker thread function to process analysis requests from the queue."""
        add_script_run_ctx(threading.current_thread())
        thread_name = threading.current_thread().name
        self.observer.on_log(f"Worker thread '{thread_name}' started.")
        
        success_count = 0
        error_count = 0
        processed_count = 0

        while (self.is_running or not self.task_queue.empty()) and not self._stop_event.is_set():
            try:
                # Wait for a task with timeout to check for stop signal occasionally
                try:
                    task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    if not self.is_running or self._stop_event.is_set():
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
                        available_sub_labels_context=context_sub_labels,
                        context_padding=context_padding,
                        overlay_config=overlay_config
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
        result_callback: Callable[[int, List[str], str], None], # Callback to save results (idx, labels, notes)
        context_padding: int = 200,
        overlay_config: Optional[Dict[str, Any]] = None
    ):
        self._stop_event.clear()
        self.is_running = True

        total = len(process_indices)
        
        self.observer.on_start(total)
        current_thread_name = threading.current_thread().name
        self.observer.on_log(f"Starting batch producer for {total} segments in thread: {current_thread_name}.")

        # Start Worker Thread
        worker_thread = threading.Thread(
            target=self._worker_process, 
            args=(result_callback, label_name_to_id, label_mapping, total, context_padding, overlay_config),
            name="GeminiWorkerThread",
            daemon=True
        )
        add_script_run_ctx(worker_thread)
        worker_thread.start()

        # Standard Logic
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
                    self.observer.on_log(f"Error deciding segment #{idx}: {str(e)}")

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

        # Always include "Segment Type" context
        if "Segment Type" in label_categories:
            other_ids = label_categories["Segment Type"]
            if other_ids:
                if "Segment Type" in main_guidelines:
                    context_list.append(f"Guideline for 'Segment Type': {main_guidelines['Segment Type']}")

                context_list.append("Available 'Segment Type' Labels:")
                for sub_id in other_ids:
                    full_name = label_mapping.get(sub_id, sub_id)
                    context_list.append(f"  - {full_name}")

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


class BatchFileJobManager:
    """
    Manages Gemini Batch API file-based jobs for segment annotation.
    Uses File API to upload JSONL requests file, then creates batch job.
    Designed for Streamlit integration with persistent state.
    """
    
    def __init__(self, gemini_analyzer, observer: BatchProgressObserver):
        self.analyzer = gemini_analyzer
        self.observer = observer
        self.state = BatchFileJobState()
    
    def prepare_and_submit(
        self,
        process_indices: List[int],
        annotations: List[Any],
        df: pd.DataFrame,
        track_config: Dict,
        label_mapping: Dict,
        label_name_to_id: Dict,
        main_guidelines: Dict,
        label_categories: Dict,
        include_graphs: bool = True,
        context_padding: int = 200,
        overlay_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Prepares batch requests and submits them as a file-based job.
        
        1. Prepares requests with keys in JSONL format
        2. Uploads JSONL file via File API
        3. Creates batch job referencing the uploaded file
        
        Args:
            include_graphs: If False, skips graph generation for text-only requests.
        
        Returns:
            True if job was submitted successfully
        """
        self.state = BatchFileJobState()
        self.state.status = "preparing"
        self.state.start_time = time.time()
        
        total = len(process_indices)
        self.observer.on_start(total)
        self.observer.on_log(f"📦 Preparing {total} requests for Batch API file job...")
        
        requests_list = []
        self.state.segment_index_map = {}
        
        for i, idx in enumerate(process_indices):
            try:
                if idx < 0 or idx >= len(annotations):
                    self.observer.on_log(f"⚠️ Invalid segment index: {idx}. Skipping.")
                    continue
                    
                ann = annotations[idx]
                segment_df = df.iloc[int(ann.start_index):int(ann.end_index)]
                
                if segment_df.empty:
                    self.observer.on_log(f"⚠️ Segment #{idx} is empty. Skipping.")
                    continue
                
                # Context slice (using context_padding)
                start_ctx = max(0, int(ann.start_index) - context_padding)
                end_ctx = min(len(df), int(ann.end_index) + context_padding)
                context_df = df.iloc[start_ctx:end_ctx]
                
                # Prepare context strings
                current_labels_display = [label_mapping.get(l, l) for l in ann.labels]
                context_sub_labels = self._build_sub_context(
                    current_labels_display, 
                    label_name_to_id, 
                    main_guidelines, 
                    label_categories,
                    label_mapping
                )
                
                # Prepare the batch request (returns {"key": "segment-{idx}", "request": {...}})
                request = self.analyzer.prepare_batch_request(
                    df=segment_df,
                    segment_index=idx,
                    track_config=track_config,
                    current_labels=current_labels_display,
                    available_sub_labels_context=context_sub_labels,
                    include_graphs=include_graphs,
                    context_df=context_df,
                    context_padding=context_padding,
                    overlay_config=overlay_config
                )
                
                if request:
                    requests_list.append(request)
                    # Map the key to segment index for result processing
                    self.state.segment_index_map[request["key"]] = idx
                    
                self.observer.on_progress(i + 1, total, f"Prepared segment #{idx}")
                
            except Exception as e:
                self.observer.on_log(f"⚠️ Error preparing segment #{idx}: {e}")
        
        if not requests_list:
            self.observer.on_error("No valid requests could be prepared.")
            self.state.status = "failed"
            return False
            
        self.state.total_requests = len(requests_list)
        graph_status = "with base64 images" if include_graphs else "text-only (no graphs)"
        self.observer.on_log(f"✅ Prepared {len(requests_list)} requests {graph_status}.")
        
        # Calculate approximate payload size
        payload_str = json.dumps(requests_list)
        payload_size_kb = len(payload_str.encode('utf-8')) / 1024
        payload_size_mb = payload_size_kb / 1024
        self.observer.on_log(f"📊 Total payload size: ~{payload_size_mb:.2f} MB ({payload_size_kb:.1f} KB)")
        
        # Submit the file-based batch job
        self.observer.on_log("📤 Uploading JSONL file and submitting batch job to Gemini API...")
        self.state.status = "uploading"
        
        job_display_name = f"acla-annotation-{int(time.time())}"
        batch_job = self.analyzer.submit_batch_file_job(
            requests_list=requests_list,
            display_name=job_display_name
        )
        
        if batch_job and hasattr(batch_job, 'name'):
            self.state.job_name = batch_job.name
            self.state.job_display_name = job_display_name
            self.state.status = "polling"
            self.observer.on_log(f"✅ Batch job submitted: {batch_job.name}")
            return True
        else:
            self.state.status = "failed"
            self.state.error_message = "Failed to create batch job"
            self.observer.on_error("Failed to create batch job. Check API key and quota.")
            return False
    
    def poll_status(self) -> Dict[str, Any]:
        """
        Polls the current job status once (non-blocking).
        
        Returns:
            Status dict with: {state, finished, success, error}
        """
        if not self.state.job_name:
            return {"state": "NO_JOB", "finished": True, "success": False}
            
        self.state.poll_count += 1
        status = self.analyzer.get_batch_job_status(self.state.job_name)
        
        self.observer.on_log(f"🔄 Poll #{self.state.poll_count}: {status['state']}")
        
        if status['finished']:
            if status['success']:
                self.state.status = "completed"
            elif status.get('error'):
                self.state.status = "failed"
                self.state.error_message = status['error']
            else:
                self.state.status = "failed" if status['state'] == 'JOB_STATE_FAILED' else "cancelled"
                
        return status
    
    def get_results(self) -> List[Dict[str, Any]]:
        """
        Retrieves and processes results from a completed file-based batch job.
        Downloads result file and maps responses back to segment indices using keys.
        
        Returns:
            List of result dicts with segment_index and parsed response
        """
        if not self.state.job_name:
            return []
            
        try:
            batch_job = self.analyzer.client.batches.get(name=self.state.job_name)
            raw_results = self.analyzer.process_batch_file_results(batch_job)
            
            processed_results = []
            for result in raw_results:
                # Map key back to segment index
                result_key = result.get("key")
                segment_idx = self.state.segment_index_map.get(result_key, -1) if result_key else -1
                
                parsed_entry = {
                    "segment_index": segment_idx,
                    "key": result_key,
                    "response_json": result.get("response_json"),
                    "error": result.get("error"),
                    "parsed_labels": [],
                    "summary": ""
                }
                
                # Try to parse JSON response
                if result.get("response_json"):
                    try:
                        raw_json = result["response_json"]
                        # Cleanup markdown if present
                        if "```json" in raw_json:
                            raw_json = raw_json.split("```json")[1].split("```")[0].strip()
                        elif "```" in raw_json:
                            raw_json = raw_json.split("```")[1].split("```")[0].strip()
                        
                        data = json.loads(raw_json)
                        parsed_entry["summary"] = data.get("analysis_summary", "")
                        parsed_entry["parsed_labels"] = data.get("suggested_labels", [])
                    except json.JSONDecodeError as e:
                        parsed_entry["error"] = f"JSON parse error: {e}"
                        
                processed_results.append(parsed_entry)
                
            self.state.results = processed_results
            self.observer.on_log(f"📥 Retrieved {len(processed_results)} results from batch job.")
            return processed_results
            
        except Exception as e:
            self.observer.on_error(f"Error retrieving batch results: {e}")
            return []
    
    def cancel_job(self) -> bool:
        """Cancels the current batch job if running."""
        if self.state.job_name and self.state.status == "polling":
            success = self.analyzer.cancel_batch_job(self.state.job_name)
            if success:
                self.state.status = "cancelled"
                self.observer.on_log("🛑 Batch job cancellation requested.")
            return success
        return False
    
    def reset(self):
        """Resets the manager state for a new job."""
        self.state = BatchFileJobState()
        self.observer.on_log("🔄 Batch job manager reset.")
    
    def _build_sub_context(self, current_labels, label_name_to_id, main_guidelines, label_categories, label_mapping):
        """Helper to build context strings for sub-labels. (Same as BatchAnnotationService)"""
        context_list = []

        # Always include "Segment Type" context
        if "Segment Type" in label_categories:
            other_ids = label_categories["Segment Type"]
            if other_ids:
                if "Segment Type" in main_guidelines:
                    context_list.append(f"Guideline for 'Segment Type': {main_guidelines['Segment Type']}")

                context_list.append("Available 'Segment Type' Labels:")
                for sub_id in other_ids:
                    full_name = label_mapping.get(sub_id, sub_id)
                    context_list.append(f"  - {full_name}")

        for lname in current_labels:
            lid = label_name_to_id.get(lname)
            if lid:
                if lid in main_guidelines:
                    desc = main_guidelines[lid]
                    context_list.append(f"Instruction for current label '{lname}': {desc}")
                
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
