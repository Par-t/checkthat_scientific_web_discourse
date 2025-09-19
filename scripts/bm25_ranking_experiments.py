#!/usr/bin/env python3
"""
BM25 Ranking experiments for all preprocessing methods.
Builds BM25 top-k datasets and saves them with appropriate naming.
"""

import os
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import json
from datetime import datetime

class BM25RankingExperiment:
    def __init__(self, data_dir="../data/checkthat"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.preprocessed_dir = self.data_dir / "preprocessed"
        self.ranked_dir = self.data_dir / "ranked"
        
        # Create ranked directory
        self.ranked_dir.mkdir(parents=True, exist_ok=True)
        
        # BM25 parameters
        self.k = 1000
        self.bm25_params = {"k1": 1.5, "b": 0.75}
        
        # Column mappings
        self.title_col = 'title_processed'
        self.abstract_col = 'abstract_processed'
        self.cord_uid_col = 'cord_uid'
        self.tweet_col = 'tweet_text_processed'
        
    def load_collection_data(self):
        """Load the collection data from pickle file"""
        print("Loading collection data...")
        with open(self.raw_dir / "subtask4b_collection_data.pkl", 'rb') as f:
            df_collection = pickle.load(f)
        print(f"Collection data shape: {df_collection.shape}")
        return df_collection
    
    def load_preprocessed_data(self, preprocessing_method):
        """Load preprocessed collection and query data"""
        print(f"Loading preprocessed data for method: {preprocessing_method}")
        
        # Load collection
        collection_file = self.preprocessed_dir / f"collection_{preprocessing_method}.csv"
        df_collection = pd.read_csv(collection_file)
        
        # Load queries
        query_file = self.preprocessed_dir / f"queries_train_{preprocessing_method}.csv"
        df_queries = pd.read_csv(query_file)
        
        
        print(f"Collection file: {collection_file}")
        print(f"Queries file   : {query_file}")
        print(f"Collection shape: {df_collection.shape}")
        print(f"Queries shape   : {df_queries.shape}")
        missing_cols = [c for c in [self.title_col, self.abstract_col, self.cord_uid_col] if c not in df_collection.columns]
        if missing_cols:
            print(f"\n[WARN] Missing expected collection columns: {missing_cols}")
        else:
            print(f"Using collection columns: title='{self.title_col}', abstract='{self.abstract_col}', id='{self.cord_uid_col}'")
        
        return df_collection, df_queries
    
    def build_corpus(self, df_collection, preprocessing_method):
        """Build corpus from collection data"""
        print(f"Building corpus for {preprocessing_method}...")
        
        # Concatenate title and abstract (processed columns) into a single corpus string
        cols = [self.title_col, self.abstract_col]
        for c in cols:
            if c not in df_collection.columns:
                raise KeyError(f"Expected column '{c}' not found in collection dataframe for method '{preprocessing_method}'")
        corpus = (
            df_collection[cols]
            .fillna('')
            .apply(lambda x: f"{x[self.title_col]} {x[self.abstract_col]}", axis=1)
            .tolist()
        )
        
        # Tokenize corpus
        tokenized_corpus = [doc.split(' ') for doc in corpus]
        cord_uids = df_collection[self.cord_uid_col].tolist()
        
        print(f"Corpus size: {len(tokenized_corpus)}")
        if len(tokenized_corpus) > 0:
            print(f"Confirmed: title + abstract concatenated (processed)")
            print(f"Sample tokenized doc: {tokenized_corpus[0][:10]}...")
        
        return tokenized_corpus, cord_uids
    
    def build_bm25_index(self, tokenized_corpus):
        """Build BM25 index from tokenized corpus"""
        print("Building BM25 index...")
        bm25 = BM25Okapi(tokenized_corpus, **self.bm25_params)
        print("BM25 index built successfully")
        return bm25
    
    def get_top_cord_uids_and_scores(self, bm25, cord_uids, query, text2bm25top):
        """Get top-k cord_uids and scores for a query"""
        if query in text2bm25top.keys():
            return text2bm25top[query]
        else:
            tokenized_query = query.split(' ')
            doc_scores = bm25.get_scores(tokenized_query)
            indices = np.argsort(-doc_scores)[:self.k]
            bm25_topk_ids = [cord_uids[x] for x in indices]
            bm25_topk_scores = [doc_scores[x] for x in indices]
            result = (bm25_topk_ids, bm25_topk_scores)
            text2bm25top[query] = result
            return result
    
    def run_ranking_experiment(self, preprocessing_method):
        """Run BM25 ranking experiment for a specific preprocessing method"""
        print(f"\n{'='*60}")
        print(f"Running BM25 ranking for: {preprocessing_method}")
        print(f"{'='*60}")
        
        # Load data
        df_collection, df_queries = self.load_preprocessed_data(preprocessing_method)
        
        # Build corpus
        tokenized_corpus, cord_uids = self.build_corpus(df_collection, preprocessing_method)
        
        # Build BM25 index
        bm25 = self.build_bm25_index(tokenized_corpus)
        
        # Prepare query data
        df_queries_work = df_queries.copy()
        
        # Use processed query text for matching; keep original for output
        query_text_col = self.tweet_col
        if query_text_col not in df_queries_work.columns:
            raise KeyError(f"Expected query column '{query_text_col}' not found in queries for method '{preprocessing_method}'")
        print(f"Query matching column: '{query_text_col}' (processed)")
        if 'tweet_text' in df_queries_work.columns and len(df_queries_work) > 0:
            _orig = str(df_queries_work['tweet_text'].iloc[0])[:120].replace("\n", " ")
            _proc = str(df_queries_work[query_text_col].iloc[0])[:120].replace("\n", " ")
            print(f"Sample original query : {_orig}")
            print(f"Sample processed query: {_proc}")
        
        # Cache for BM25 results
        text2bm25top = {}
        
        # Enable tqdm for pandas apply
        tqdm.pandas(desc=f"Processing BM25 Queries - {preprocessing_method}")
        
        # Apply BM25 ranking with progress bar
        bm25_results = df_queries_work[query_text_col].progress_apply(
            lambda x: self.get_top_cord_uids_and_scores(bm25, cord_uids, x, text2bm25top)
        )
        
        # Split results into separate columns
        df_queries_work['bm25_topk'] = [result[0] for result in bm25_results]
        df_queries_work['scores'] = [result[1] for result in bm25_results]
        
        # Create results dataframe with only essential columns
        results_df = df_queries_work[['post_id', 'cord_uid', 'tweet_text', 'bm25_topk', 'scores']].copy()
        
        # Rename columns for clarity
        results_df = results_df.rename(columns={
            'tweet_text': 'query'
        })
        
        # Save results
        output_file = self.ranked_dir / f"bm25_topk_{preprocessing_method}.csv"
        results_df.to_csv(output_file, index=False)
        
        print(f"✓ Saved BM25 rankings: {output_file}")
        print(f"  Shape: {results_df.shape}")
        print(f"  Sample top-k length: {len(results_df['bm25_topk'].iloc[0])}")
        print(f"  Sample scores length: {len(results_df['scores'].iloc[0])}")
        
        return results_df
    
    def run_all_experiments(self):
        """Run BM25 ranking experiments for all preprocessing methods"""
        print("Starting BM25 ranking experiments...")
        
        
        preprocessing_methods = [
            "baseline",
            "stopwords_removed", 
            "stopwords_punc_removed",
            "stopwords_punc_stemmed",
            "stopwords_punc_lemmatized"
        ]
        
        # Check if preprocessed data exists
        missing_methods = []
        for method in preprocessing_methods:
            collection_file = self.preprocessed_dir / f"collection_{method}.csv"
            query_file = self.preprocessed_dir / f"queries_train_{method}.csv"
            
            if not collection_file.exists() or not query_file.exists():
                missing_methods.append(method)
        
        if missing_methods:
            print(f"Warning: Missing preprocessed data for: {missing_methods}")
            print("Please run preprocessing experiments first.")
            return
        
        # Run experiments
        all_results = {}
        for method in preprocessing_methods:
            try:
                results_df = self.run_ranking_experiment(method)
                all_results[method] = results_df
            except Exception as e:
                print(f"Error processing {method}: {e}")
                continue
        
        # Create summary
        self.create_experiment_summary(all_results)
        
        print(f"\n{'='*60}")
        print("All BM25 ranking experiments completed!")
        print(f"Results saved in: {self.ranked_dir}")
        print(f"{'='*60}")
    
    def create_experiment_summary(self, all_results):
        """Create summary of all experiments"""
        summary = {
            "experiment_type": "bm25_ranking",
            "timestamp": datetime.now().isoformat(),
            "bm25_params": self.bm25_params,
            "k": self.k,
            "preprocessing_methods": list(all_results.keys()),
            "results": {}
        }
        
        for method, df in all_results.items():
            summary["results"][method] = {
                "shape": df.shape,
                "sample_topk_length": len(df['bm25_topk'].iloc[0]) if len(df) > 0 else 0,
                "output_file": f"bm25_topk_{method}.csv"
            }
        
        # Save summary
        summary_file = self.ranked_dir / "bm25_experiments_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary saved: {summary_file}")

def main():
    """Main function"""
    experiment = BM25RankingExperiment()
    experiment.run_all_experiments()

if __name__ == "__main__":
    main()
