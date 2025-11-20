import streamlit as st
import os
import json
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add parent directory to path to allow importing app modules
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.services.hf_cloud_llm_service import HuggingFaceCloudLLM

# Default directory where datasets are written
# Assuming this file is in acla_ai_service/ui/
DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[1] / "models" / "llm_datasets"

def load_dataset_stats(file_path: Path) -> Dict[str, Any]:
    total = 0
    annotated = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                    total += 1
                    # Check for annotation completion marker
                    # Based on telemetry_prompt_annotation_app.py logic
                    metadata = data.get("metadata", {})
                    if isinstance(metadata, dict) and metadata.get("annotation_complete"):
                        annotated += 1
                    elif data.get("annotation_complete"): # Fallback
                        annotated += 1
                except:
                    pass
    except Exception as e:
        return {"error": str(e)}
    
    return {
        "total_examples": total,
        "annotated_examples": annotated,
        "path": str(file_path)
    }

def main():
    st.set_page_config(page_title="LLM Training Manager", layout="wide")
    st.title("LLM Training & Fine-tuning Manager")
    
    st.sidebar.header("Configuration")
    dataset_dir_str = st.sidebar.text_input("Dataset Directory", str(DEFAULT_DATASET_DIR))
    dataset_dir = Path(dataset_dir_str)
    
    if not dataset_dir.exists():
        st.warning(f"Directory not found: {dataset_dir}. Creating it...")
        try:
            dataset_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            st.error(f"Failed to create directory: {e}")
            return

    st.header("1. Select Datasets")
    
    # Find jsonl files
    dataset_files = sorted(list(dataset_dir.rglob("*.jsonl")))
    # Filter out annotation stores if they follow a naming convention (e.g. .annotations.jsonl)
    dataset_files = [f for f in dataset_files if not f.name.endswith(".annotations.jsonl")]
    
    selected_datasets = []
    
    if not dataset_files:
        st.info("No datasets found.")
    else:
        # Create a table of datasets
        st.write("Available Datasets:")
        
        # Header
        c1, c2, c3, c4 = st.columns([0.5, 4, 2, 2])
        c1.write("**Select**")
        c2.write("**Filename**")
        c3.write("**Progress (Annotated/Total)**")
        c4.write("**Status**")
        
        for f in dataset_files:
            stats = load_dataset_stats(f)
            if "error" in stats:
                continue
            
            col1, col2, col3, col4 = st.columns([0.5, 4, 2, 2])
            with col1:
                if st.checkbox("", key=str(f)):
                    selected_datasets.append(str(f))
            with col2:
                st.text(f.name)
            with col3:
                st.text(f"{stats['annotated_examples']}/{stats['total_examples']}")
            with col4:
                if stats['annotated_examples'] > 0:
                    st.success("Ready")
                else:
                    st.warning("Pending")

    st.header("2. Training Actions")
    
    if not selected_datasets:
        st.warning("Please select at least one dataset to proceed.")
    else:
        st.write(f"Selected {len(selected_datasets)} datasets.")
        
        col_local, col_hf = st.columns(2)
        
        with col_local:
            st.subheader("Local Training")
            epochs = st.number_input("Epochs", min_value=1, value=3)
            batch_size = st.number_input("Batch Size", min_value=1, value=4)
            learning_rate = st.number_input("Learning Rate", value=2e-5, format="%.1e")
            
            if st.button("Start Local Training"):
                st.info("Starting local training... (This is a placeholder)")
                # TODO: Call actual training logic
                # from app.services.llm.trainer import train_local
                # train_local(selected_datasets, epochs, batch_size, learning_rate)
                
        with col_hf:
            st.subheader("Upload to Hugging Face")
            hf_repo = st.text_input("HF Repository ID (Optional)", value="", help="Leave empty to automatically create a new dataset repository.")
            
            if st.button("Upload Datasets"):
                st.info("Uploading to Hugging Face...")
                
                # Merge datasets
                merged_file_path = dataset_dir / "merged_training_data.jsonl"
                try:
                    with open(merged_file_path, 'w', encoding='utf-8') as outfile:
                        for fname in selected_datasets:
                            with open(fname, 'r', encoding='utf-8') as infile:
                                for line in infile:
                                    outfile.write(line)
                    
                    # Initialize service
                    service = HuggingFaceCloudLLM()
                    
                    # Upload
                    # We need a dummy output_dir for the train method signature
                    dummy_output = dataset_dir / "hf_upload_output"
                    
                    result = service.train(
                        dataset_path=merged_file_path,
                        output_dir=dummy_output,
                        repo_id=hf_repo if hf_repo.strip() else None
                    )
                    
                    st.success(f"Upload complete! Dataset available at: {result['cloud_training_info']['dataset_url']}")
                    st.json(result['cloud_training_info'])
                    
                except Exception as e:
                    st.error(f"Upload failed: {str(e)}")
                finally:
                    # Cleanup merged file
                    if merged_file_path.exists():
                        try:
                            os.remove(merged_file_path)
                        except:
                            pass

if __name__ == "__main__":
    main()
