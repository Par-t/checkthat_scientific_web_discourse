#!/usr/bin/env python3
"""
Preprocessing experiments for BM25 comparison.
Generates 5 different preprocessing variants and saves them with clear naming.
"""
    
import os
import pandas as pd
import pickle
import string
import re
import unicodedata
from pathlib import Path
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text):
        """Clean text by removing emojis, URLs, mentions, hashtags - not punctuation"""
        if pd.isna(text) or text == '':
            return text
            
        # Remove emojis and special unicode characters
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if not unicodedata.combining(c))
        
        # Remove emojis using regex
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"  # dingbats
            u"\U000024C2-\U0001F251"  # enclosed characters
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Clean up multiple spaces (but keep punctuation)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
        
    def preprocess_baseline(self, text):
        """No preprocessing - just return as is"""
        return text
    
    def preprocess_stopwords(self, text):
        """Remove stopwords only (lowercase, whitespace tokenization)"""
        if pd.isna(text) or text == '':
            return text
            
        # Clean text first
        text = self.clean_text(text)

        # Whitespace tokenize, lowercase, remove stopwords
        tokens = str(text).lower().split()
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        return ' '.join(filtered_tokens)
    
    def preprocess_stopwords_punc(self, text):
        """Remove stopwords then punctuation (no stemming/lemmatization)"""
        if pd.isna(text) or text == '':
            return text
            
        # Clean text first
        text = self.clean_text(text)

        # Remove stopwords
        tokens = str(text).lower().split()
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        text_no_sw = ' '.join(filtered_tokens)

        # Remove punctuation
        translator = str.maketrans('', '', string.punctuation)
        return text_no_sw.translate(translator)
    
    def preprocess_stopwords_punc_stem(self, text):
        """Remove stopwords, apply stemming, then remove punctuation"""
        if pd.isna(text) or text == '':
            return text
            
        # Clean text first
        text = self.clean_text(text)

        # Remove stopwords
        tokens = str(text).lower().split()
        filtered_tokens = [word for word in tokens if word not in self.stop_words]

        # Stem
        stemmed_tokens = [self.stemmer.stem(word) for word in filtered_tokens]
        text_stemmed = ' '.join(stemmed_tokens)

        # Remove punctuation
        translator = str.maketrans('', '', string.punctuation)
        return text_stemmed.translate(translator)
    
    def preprocess_stopwords_punc_lemma(self, text):
        """Remove stopwords, apply lemmatization, then remove punctuation"""
        if pd.isna(text) or text == '':
            return text
            
        # Clean text first
        text = self.clean_text(text)

        # Remove stopwords
        tokens = str(text).lower().split()
        filtered_tokens = [word for word in tokens if word not in self.stop_words]

        # Lemmatize
        lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in filtered_tokens]
        text_lemma = ' '.join(lemmatized_tokens)

        # Remove punctuation
        translator = str.maketrans('', '', string.punctuation)
        return text_lemma.translate(translator)

def load_data():
    """Load the raw data"""
    print("Loading raw data...")
    
    # Load collection data
    with open("../data/checkthat/raw/subtask4b_collection_data.pkl", 'rb') as f:
        df_collection = pickle.load(f)
    
    # Load only train data
    df_query_train = pd.read_csv("../data/checkthat/raw/subtask4b_query_tweets_train.tsv", sep='\t')
    
    print(f"Collection data: {df_collection.shape}")
    print(f"Query train: {df_query_train.shape}")
    
    return df_collection, df_query_train

def clean_and_deduplicate_dataframe(df, file_type="collection"):
    """
    Clean and deduplicate a dataframe based on file type
    
    Args:
        df: DataFrame to clean
        file_type: "collection" or "queries"
    
    Returns:
        Cleaned DataFrame
    """
    original_shape = df.shape
    
    if file_type == "collection":
        # For collection files, dedupe by cord_uid
        
        df_cleaned = df.drop_duplicates(subset=['cord_uid'], keep='first')
        dedup_col = 'cord_uid'

        
    elif file_type == "queries":
        # For query files, dedupe by tweet_text
        df_cleaned = df.drop_duplicates(subset=['tweet_text'], keep='first')
        dedup_col = 'tweet_text'
 
    else:
        print(f"Unknown file type: {file_type}")
    
    duplicates_removed = original_shape[0] - df_cleaned.shape[0]
    
    if duplicates_removed > 0:
        print(f"  ⚠️  Removed {duplicates_removed} duplicates (based on {dedup_col})")
    else:
        print(f"  ✓ No duplicates found")
    
    return df_cleaned

def preprocess_dataframe(df, text_column, preprocessor, method_name, progress_desc, file_type="collection"):
    """Apply preprocessing to a dataframe column with deduplication"""
    print(f"\n{progress_desc}...")
    
    # Create a copy to avoid modifying original
    df_processed = df.copy()
    
    # Apply preprocessing with progress bar
    df_processed[f'{text_column}_processed'] = [
        preprocessor(text) for text in tqdm(df[text_column], desc=progress_desc)
    ]
    
    # Add original text backup
    df_processed[f'original_{text_column}'] = df[text_column]
    
    # Clean and deduplicate
    print(f"Cleaning and deduplicating {file_type} data...")
    df_processed = clean_and_deduplicate_dataframe(df_processed, file_type)
    
    return df_processed

def save_preprocessed_data(df_collection, df_query_train, method_name, output_dir):
    """Save preprocessed data with clear naming convention"""
    
    # Collection data - only essential columns
    collection_file = output_dir / f"collection_{method_name}.csv"
    df_collection_essential = df_collection[['cord_uid', 'title_processed', 'abstract_processed']]
    df_collection_essential.to_csv(collection_file, index=False)
    print(f"Saved collection data: {collection_file}")
    print(f"  Columns: {df_collection_essential.columns.tolist()}")
    
    # Query data - only essential columns
    query_train_file = output_dir / f"queries_train_{method_name}.csv"
    df_query_essential = df_query_train[['post_id', 'cord_uid', 'tweet_text', 'tweet_text_processed']]
    df_query_essential.to_csv(query_train_file, index=False)
    
    print(f"Saved query data: {query_train_file}")
    print(f"  Columns: {df_query_essential.columns.tolist()}")

def main():
    """Main preprocessing pipeline"""
    
    # Create output directory
    output_dir = Path("../data/checkthat/preprocessed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load raw data
    df_collection, df_query_train = load_data()
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Define preprocessing methods
    preprocessing_methods = [
        {
            'name': 'baseline',
            'description': 'No preprocessing',
            'function': preprocessor.preprocess_baseline
        },
        {
            'name': 'stopwords_removed',
            'description': 'Stop words removed',
            'function': preprocessor.preprocess_stopwords
        },
        {
            'name': 'stopwords_punc_removed',
            'description': 'Stop words and punctuation removed',
            'function': preprocessor.preprocess_stopwords_punc
        },
        {
            'name': 'stopwords_punc_stemmed',
            'description': 'Stop words + punctuation removed + stemmed',
            'function': preprocessor.preprocess_stopwords_punc_stem
        },
        {
            'name': 'stopwords_punc_lemmatized',
            'description': 'Stop words + punctuation removed + lemmatized',
            'function': preprocessor.preprocess_stopwords_punc_lemma
        }
    ]
    
    # Process each method
    for method in preprocessing_methods:
        print(f"\n{'='*60}")
        print(f"Processing: {method['description']}")
        print(f"{'='*60}")
        
        # Process collection data (both 'abstract' and 'title' columns)
        print(f"Processing collection abstracts...")
        df_collection_processed = df_collection.copy()
        
        # Process abstracts
        df_collection_processed['abstract_processed'] = [
            method['function'](text) for text in tqdm(df_collection['abstract'], 
                                                   desc=f"Collection abstracts - {method['description']}")
        ]
        
        # Process titles
        df_collection_processed['title_processed'] = [
            method['function'](text) for text in tqdm(df_collection['title'], 
                                                   desc=f"Collection titles - {method['description']}")
        ]
        
        # Clean and deduplicate collection data
        print(f"Cleaning and deduplicating collection data...")
        df_collection_processed = clean_and_deduplicate_dataframe(df_collection_processed, "collection")
        
        # Process query data (using 'tweet_text' column)
        df_query_train_processed = preprocess_dataframe(
            df_query_train,
            'tweet_text',
            method['function'],
            method['name'],
            f"Train queries - {method['description']}",
            file_type="queries"
        )
        
        # Save processed data
        save_preprocessed_data(
            df_collection_processed,
            df_query_train_processed,
            method['name'],
            output_dir
        )
        
        print(f"✓ Completed: {method['description']}")
    
    print(f"\n{'='*60}")
    print("All preprocessing experiments completed!")
    print(f"Results saved in: {output_dir}")
    print(f"{'='*60}")
    
    # Create a summary file
    summary_file = output_dir / "preprocessing_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Preprocessing Experiments Summary\n")
        f.write("=" * 40 + "\n\n")
        for method in preprocessing_methods:
            f.write(f"{method['name']}: {method['description']}\n")
        f.write(f"\nTotal methods: {len(preprocessing_methods)}\n")
        f.write(f"Output directory: {output_dir}\n")
    
    print(f"Summary saved: {summary_file}")

if __name__ == "__main__":
    main()
