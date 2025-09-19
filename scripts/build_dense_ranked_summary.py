#!/usr/bin/env python3
"""
Build dense_experiments.json for dense retrieval experiments.

This script adds a single dense retrieval experiment to the dense_experiments.json file.

Usage:
    python scripts/build_dense_ranked_summary.py snowflake_topk.csv
"""

import json
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

class DenseRankedSummaryBuilder:
    def __init__(self, data_dir="../data/checkthat"):
        self.data_dir = Path(data_dir)
        self.ranked_dir = self.data_dir / "ranked"
        
        # Create directories if they don't exist
        self.ranked_dir.mkdir(parents=True, exist_ok=True)
        
        # Model metadata registry - add new models here
        self.model_registry = {
            "snowflake": {
                "model": "Snowflake/snowflake-arctic-embed-l",
                "link": "https://huggingface.co/Snowflake/snowflake-arctic-embed-l",
                "embeddings": 1024,
                "size": "335M",
                "max_seq_length": 512
            },
            # Add more models here as needed
            # "model_name": {
            #     "model": "Model/name",
            #     "link": "https://huggingface.co/Model/name",
            #     "embeddings": 768,
            #     "size": "110M",
            #     "max_seq_length": 512
            # }
        }
    
    def load_existing_experiments(self) -> Dict[str, Any]:
        """Load existing dense experiments from JSON file"""
        experiments_file = self.ranked_dir / "dense_experiments_summary.json"
        
        if experiments_file.exists():
            with open(experiments_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "experiment_metadata": {
                    "creation_timestamp": datetime.now().isoformat(),
                    "format_version": "1.0",
                    "description": "Dense retrieval experiments summary"
                },
                "experiments": {}
            }
    
    def add_experiment(self, csv_filename: str) -> Dict[str, Any]:
        """Add a single experiment to the dense experiments JSON"""
        
        # Load existing experiments
        experiments_data = self.load_existing_experiments()
        
        # Check if CSV file exists
        csv_path = self.ranked_dir / csv_filename
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Extract model name from filename (e.g., "snowflake_topk.csv" -> "snowflake")
        model_name = csv_filename.replace("_topk.csv", "")
        
        # Get model metadata from registry
        if model_name not in self.model_registry:
            raise ValueError(f"Model '{model_name}' not found in registry. Available models: {list(self.model_registry.keys())}")
        
        model_metadata = self.model_registry[model_name]
        
        # Read CSV to get basic info
        try:
            df = pd.read_csv(csv_path, nrows=5)
            
            # Create experiment entry in your preferred format
            experiment_data = {
                "model": model_metadata["model"],
                "output_file": csv_filename,
                "link": model_metadata["link"],
                "embeddings": model_metadata["embeddings"],
                "size": model_metadata["size"],
                "max_seq_length": model_metadata["max_seq_length"],
                "added_at": datetime.now().isoformat(),
                "file_size_bytes": csv_path.stat().st_size,
                "columns": list(df.columns)
            }
            
            # Add to experiments
            experiments_data["experiments"][model_name] = experiment_data
            experiments_data["experiment_metadata"]["last_updated"] = datetime.now().isoformat()
            experiments_data["experiment_metadata"]["total_experiments"] = len(experiments_data["experiments"])
            
            return experiments_data
            
        except Exception as e:
            raise Exception(f"Could not read {csv_path}: {e}")
    
    def save_experiments(self, experiments_data: Dict[str, Any]) -> Path:
        """Save experiments to JSON file"""
        output_path = self.ranked_dir / "dense_experiments_summary.json"
        
        with open(output_path, 'w') as f:
            json.dump(experiments_data, f, indent=2)
        
        print(f"Dense experiments summary saved to: {output_path}")
        return output_path
    
    def print_experiment_info(self, experiments_data: Dict[str, Any], model_name: str):
        """Print information about the added experiment"""
        print("\n" + "="*60)
        print("DENSE RETRIEVAL EXPERIMENT ADDED")
        print("="*60)
        
        if model_name in experiments_data["experiments"]:
            exp_data = experiments_data["experiments"][model_name]
            print(f"Model: {exp_data['model']}")
            print(f"Output file: {exp_data['output_file']}")
            print(f"Link: {exp_data['link']}")
            print(f"Embeddings: {exp_data['embeddings']}")
            print(f"Size: {exp_data['size']}")
            print(f"Max sequence length: {exp_data['max_seq_length']}")
            print(f"File size: {exp_data['file_size_bytes']:,} bytes")
            print(f"Columns: {exp_data['columns']}")
            print(f"Added at: {exp_data['added_at']}")
        
        print(f"\nTotal experiments: {experiments_data['experiment_metadata']['total_experiments']}")
        print("="*60)

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python scripts/build_dense_ranked_summary.py <csv_filename>")
        print("Example: python scripts/build_dense_ranked_summary.py snowflake_topk.csv")
        sys.exit(1)
    
    csv_filename = sys.argv[1]
    builder = DenseRankedSummaryBuilder()
    
    try:
        print(f"Adding dense retrieval experiment: {csv_filename}")
        
        # Add experiment
        experiments_data = builder.add_experiment(csv_filename)
        
        # Save experiments
        output_path = builder.save_experiments(experiments_data)
        
        # Print experiment info
        model_name = csv_filename.replace("_topk.csv", "")
        builder.print_experiment_info(experiments_data, model_name)
        
        print(f"\nExperiment added successfully!")
        print(f"Saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
