#!/usr/bin/env python3
"""
Build ranked_evaluation.json by reading both BM25 and dense experiment summaries.

- Reads: data/checkthat/ranked/bm25_experiments_summary.json
- Reads: data/checkthat/ranked/dense_experiments_summary.json  
- Evaluates: all listed experiments for k in [5, 10, 25]
- Writes: experimental_results/ranked_evaluation.json

Preserves all original parameters and information, and adds metrics fields.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import numpy as np
import ast
from tqdm import tqdm
from datetime import datetime


def _safe_parse_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        v = value.strip()
        # Try JSON first
        try:
            parsed = json.loads(v)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            pass
        # Fallback to Python literal
        try:
            parsed = ast.literal_eval(v)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def compute_ndcg_at_k(y_true: List[str], y_pred: List[str], k: int) -> float:
    if not y_pred:
        return 0.0
    y_pred = y_pred[:k]
    relevance = [1 if doc in y_pred else 0 for doc in y_true]
    dcg = 0.0
    for i, rel in enumerate(relevance):
        if rel > 0:
            dcg += rel / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(y_true), k)))
    return float(dcg / idcg) if idcg > 0 else 0.0


def compute_map_at_k(y_true: List[str], y_pred: List[str], k: int) -> float:
    if not y_pred:
        return 0.0
    y_pred = y_pred[:k]
    relevant_positions = [i + 1 for i, doc in enumerate(y_pred) if doc in y_true]
    if not relevant_positions:
        return 0.0
    precision_sum = 0.0
    for i, pos in enumerate(relevant_positions):
        precision_sum += (i + 1) / pos
    return float(precision_sum / len(y_true))


def compute_mrr(y_true: List[str], y_pred: List[str]) -> float:
    if not y_pred:
        return 0.0
    for i, doc in enumerate(y_pred):
        if doc in y_true:
            return float(1.0 / (i + 1))
    return 0.0


def compute_recall_at_k(y_true: List[str], y_pred: List[str], k: int) -> float:
    if not y_true:
        return 0.0
    y_pred = y_pred[:k]
    relevant_found = sum(1 for doc in y_pred if doc in y_true)
    return float(relevant_found / len(y_true))


def evaluate_file(filepath: Path, k_values: List[int]) -> Dict[str, Any]:
    df = pd.read_csv(filepath)

    # Find the column ending with '_topk' (could be bm25_topk, snowflake_topk, etc.)
    topk_col = None
    for col in df.columns:
        if col.endswith('_topk'):
            topk_col = col
            break
    
    if topk_col is None:
        raise ValueError(f"No column ending with '_topk' found in {filepath}. Available columns: {list(df.columns)}")

    mrr_scores: List[float] = []
    mrr_at_scores: Dict[int, List[float]] = {k: [] for k in k_values}
    ndcg_scores: Dict[int, List[float]] = {k: [] for k in k_values}
    map_scores: Dict[int, List[float]] = {k: [] for k in k_values}
    recall_scores: Dict[int, List[float]] = {k: [] for k in k_values}

    total_queries = len(df)
    queries_with_results = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {filepath.name}"):
        y_true = [row['cord_uid']] if 'cord_uid' in row and pd.notna(row['cord_uid']) else []
        y_pred = _safe_parse_list(row.get(topk_col, []))

        if y_pred:
            queries_with_results += 1
        if not y_true:
            continue

        mrr_scores.append(compute_mrr(y_true, y_pred))
        for k in k_values:
            # MRR@k via truncation
            mrr_at_scores[k].append(compute_mrr(y_true, y_pred[:k]))
            ndcg_scores[k].append(compute_ndcg_at_k(y_true, y_pred, k))
            map_scores[k].append(compute_map_at_k(y_true, y_pred, k))
            recall_scores[k].append(compute_recall_at_k(y_true, y_pred, k))

    metrics = {
        'mrr': float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        'mrr_at': {f'mrr_{k}': float(np.mean(v)) if v else 0.0 for k, v in mrr_at_scores.items()},
        'ndcg': {f'ndcg_{k}': float(np.mean(v)) if v else 0.0 for k, v in ndcg_scores.items()},
        'map': {f'map_{k}': float(np.mean(v)) if v else 0.0 for k, v in map_scores.items()},
        'recall': {f'recall_{k}': float(np.mean(v)) if v else 0.0 for k, v in recall_scores.items()},
        'total_queries': int(total_queries),
        'queries_with_results': int(queries_with_results),
        'topk_column_used': topk_col,  # Track which column was used for evaluation
    }

    return metrics


def load_experiment_summaries(ranked_dir: Path) -> Dict[str, Any]:
    """Load both BM25 and dense experiment summaries"""
    summaries = {}
    
    # Load BM25 summary
    bm25_summary_path = ranked_dir / 'bm25_experiments_summary.json'
    if bm25_summary_path.exists():
        with open(bm25_summary_path, 'r') as f:
            summaries['bm25'] = json.load(f)
        print(f"Loaded BM25 summary: {len(summaries['bm25'].get('results', {}))} experiments")
    else:
        print("Warning: BM25 summary not found")
        summaries['bm25'] = {}
    
    # Load dense summary
    dense_summary_path = ranked_dir / 'dense_experiments_summary.json'
    if dense_summary_path.exists():
        with open(dense_summary_path, 'r') as f:
            summaries['dense'] = json.load(f)
        print(f"Loaded dense summary: {len(summaries['dense'].get('experiments', {}))} experiments")
    else:
        print("Warning: Dense summary not found")
        summaries['dense'] = {}
    
    return summaries


def build_unified_evaluation_data(summaries: Dict[str, Any]) -> Dict[str, Any]:
    """Build unified evaluation data structure from both summaries"""
    
    evaluation_data = {
        "evaluation_metadata": {
            "creation_timestamp": datetime.now().isoformat(),
            "format_version": "2.0",
            "description": "Unified ranked retrieval evaluation results",
            "k_values_evaluated": [5, 10, 25],
            "metrics_computed": ["ndcg", "map", "mrr", "recall"]
        },
        "sparse": {
            "model_type": "bm25",
            "experiments": {}
        },
        "dense": {
            "model_type": "sentence-transformers", 
            "experiments": {}
        }
    }
    
    # Process BM25 experiments
    bm25_data = summaries.get('bm25', {})
    bm25_params = bm25_data.get('bm25_params', {})
    preprocessing_methods = bm25_data.get('preprocessing_methods', [])
    
    for method, data in bm25_data.get('results', {}).items():
        evaluation_data["sparse"]["experiments"][method] = {
            "model": "BM25",
            "preprocessing_method": method,
            "model_parameters": bm25_params,
            "k": bm25_data.get('k', 1000),
            "output_file": data.get('output_file', f"bm25_topk_{method}.csv"),
            "shape": data.get('shape', []),
            "sample_topk_length": data.get('sample_topk_length', 1000),
            "experiment_metadata": {
                "experiment_type": "bm25_ranking",
                "timestamp": bm25_data.get('timestamp', ''),
                "preprocessing_methods": preprocessing_methods
            }
        }
    
    # Process dense experiments
    dense_data = summaries.get('dense', {})
    for model_name, data in dense_data.get('experiments', {}).items():
        evaluation_data["dense"]["experiments"][model_name] = {
            "model": data.get('model', ''),
            "preprocessing_method": "none",  # Dense models typically don't use text preprocessing
            "model_parameters": {
                "embeddings": data.get('embeddings', 0),
                "size": data.get('size', ''),
                "max_seq_length": data.get('max_seq_length', 0),
                "link": data.get('link', '')
            },
            "output_file": data.get('output_file', ''),
            "file_size_bytes": data.get('file_size_bytes', 0),
            "columns": data.get('columns', []),
            "experiment_metadata": {
                "experiment_type": "dense_ranking",
                "added_at": data.get('added_at', ''),
                "model_link": data.get('link', '')
            }
        }
    
    return evaluation_data


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    ranked_dir = project_root / 'data' / 'checkthat' / 'ranked'
    results_dir = project_root / 'experimental_results' / 'checkthat'
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading experiment summaries...")
    summaries = load_experiment_summaries(ranked_dir)
    
    print("Building unified evaluation data...")
    evaluation_data = build_unified_evaluation_data(summaries)

    # Evaluate all experiments
    k_values = [5, 10, 25]
    
    # Evaluate sparse (BM25) experiments
    print("\nEvaluating sparse (BM25) experiments...")
    for method_name, exp_data in tqdm(evaluation_data["sparse"]["experiments"].items(), desc="BM25 experiments", unit="exp"):
        output_file = exp_data.get('output_file')
        if not output_file:
            continue
        csv_path = ranked_dir / output_file
        if not csv_path.exists():
            exp_data['metrics'] = {'error': f'File not found: {str(csv_path)}'}
            continue
        try:
            metrics = evaluate_file(csv_path, k_values)
            exp_data['metrics'] = metrics
        except Exception as e:
            exp_data['metrics'] = {'error': str(e)}
    
    # Evaluate dense experiments
    print("\nEvaluating dense experiments...")
    for model_name, exp_data in tqdm(evaluation_data["dense"]["experiments"].items(), desc="Dense experiments", unit="exp"):
        output_file = exp_data.get('output_file')
        if not output_file:
            continue
        csv_path = ranked_dir / output_file
        if not csv_path.exists():
            exp_data['metrics'] = {'error': f'File not found: {str(csv_path)}'}
            continue
        try:
            metrics = evaluate_file(csv_path, k_values)
            exp_data['metrics'] = metrics
        except Exception as e:
            exp_data['metrics'] = {'error': str(e)}

    # Save evaluation results
    output_path = results_dir / 'ranked_evaluation.json'
    with open(output_path, 'w') as f:
        json.dump(evaluation_data, f, indent=2)

    print(f"\nranked_evaluation.json saved to: {output_path}")
    
    # Print summary
    print(f"\nEvaluation Summary:")
    print(f"  Sparse experiments: {len(evaluation_data['sparse']['experiments'])}")
    print(f"  Dense experiments: {len(evaluation_data['dense']['experiments'])}")
    print(f"  Total experiments: {len(evaluation_data['sparse']['experiments']) + len(evaluation_data['dense']['experiments'])}")


if __name__ == '__main__':
    main()