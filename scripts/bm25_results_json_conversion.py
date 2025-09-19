#!/usr/bin/env python3
"""
Convert existing bm25_experiments_summary.json to new extensible format.
Creates ranked_data_summary.json with support for future model types.
"""

import json
from pathlib import Path
from datetime import datetime

def convert_bm25_summary_to_extensible():
    """Convert BM25 summary to new extensible format"""
    
    # Paths
    input_file = Path("../data/checkthat/ranked/bm25_experiments_summary.json")
    output_file = Path("../data/checkthat/ranked/ranked_data_summary.json")
    
    # Load existing BM25 summary
    with open(input_file, 'r') as f:
        bm25_data = json.load(f)
    
    # Extract BM25 parameters and preprocessing methods
    bm25_params = bm25_data.get("bm25_params", {"k1": 1.5, "b": 0.75})
    k_value = bm25_data.get("k", 1000)
    preprocessing_methods = bm25_data.get("preprocessing_methods", [])
    
    # Create new extensible structure - only what exists now
    new_structure = {
        "experiment_metadata": {
            "conversion_timestamp": datetime.now().isoformat(),
            "original_file": "bm25_experiments_summary.json",
            "format_version": "2.0"
        },
        "sparse": {
            "model": "bm25",
            "parameters": {
                **bm25_params,
                "k": k_value
            },
            "preprocessing_methods": preprocessing_methods,
            "experiments": {}
        }
    }
    
    # Convert BM25 results to new format
    if "results" in bm25_data:
        for method, result_data in bm25_data["results"].items():
            new_structure["sparse"]["experiments"][method] = {
                "output_file": result_data.get("output_file", f"bm25_topk_{method}.csv"),
                "shape": result_data.get("shape", []),
                "sample_topk_length": result_data.get("sample_topk_length", 1000),
                "status": "completed"
            }
    
    # Save new structure
    with open(output_file, 'w') as f:
        json.dump(new_structure, f, indent=2)
    
    print(f"✓ Converted {input_file} to {output_file}")
    print(f"✓ New format is extensible for future model types")
    print(f"✓ BM25 data migrated to 'sparse' section")
    print(f"✓ Ready to add 'dense' and 'fusion' sections when needed")
    
    return new_structure

if __name__ == "__main__":
    convert_bm25_summary_to_extensible()