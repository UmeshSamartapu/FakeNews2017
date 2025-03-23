#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fake news detection
The Doc2Vec pre-processing
"""

import numpy as np
import re
import string
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import logging  # Added for better logging

# Initialize logger
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def textClean(text):
    """
    Get rid of non-alphanumeric characters and normalize text
    """
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = text.lower().split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if w not in stops]
    return " ".join(text)

def cleanup(text):
    """Clean and preprocess text"""
    text = textClean(text)
    return text.translate(str.maketrans("", "", string.punctuation))

def constructLabeledSentences(data):
    """Create TaggedDocument objects for Doc2Vec training"""
    return [
        TaggedDocument(simple_preprocess(row), [f'Text_{idx}'])
        for idx, row in data.items()
    ]

def getEmbeddings(path, vector_dimension=300):
    """Generate Doc2Vec embeddings and split data"""
    # Load and clean data
    data = pd.read_csv(path)
    
    # Handle missing values more efficiently
    data = data.dropna(subset=['text']).reset_index(drop=True)
    if 'id' in data.columns:
        data = data.drop(columns=['id'])
    
    # Preprocess text
    data['cleaned_text'] = data['text'].apply(cleanup)
    
    # Create tagged documents
    tagged_data = constructLabeledSentences(data['cleaned_text'])
    
    # Train Doc2Vec model
    model = Doc2Vec(
        vector_size=vector_dimension,
        min_count=2,  # Increased from 1 to ignore rare words
        window=5,
        sample=1e-5,
        negative=5,
        workers=7,
        epochs=20,  # Increased epochs for better training
        dm=1  # Added document model type parameter
    )
    
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    
    # Generate embeddings
    train_size = int(0.8 * len(tagged_data))
    
    train_arrays = np.array([model.dv[f'Text_{i}'] for i in range(train_size)])
    test_arrays = np.array([model.dv[f'Text_{i}'] for i in range(train_size, len(tagged_data))])
    
    labels = data['label'].values
    train_labels = labels[:train_size]
    test_labels = labels[train_size:]
    
    return train_arrays, test_arrays, train_labels, test_labels

def clean_data():
    """Clean and save processed data"""
    data = pd.read_csv('datasets/train.csv')
    
    # Handle missing values
    data = data.dropna(subset=['text']).reset_index(drop=True)
    
    # Clean text
    data['text'] = data['text'].apply(cleanup)
    
    # Shuffle and split
    data = data.sample(frac=1).reset_index(drop=True)
    split_idx = int(0.8 * len(data))
    
    # Save processed data
    np.save('xtr_shuffled.npy', data['text'].values[:split_idx])
    np.save('xte_shuffled.npy', data['text'].values[split_idx:])
    np.save('ytr_shuffled.npy', data['label'].values[:split_idx])
    np.save('yte_shuffled.npy', data['label'].values[split_idx:])

if __name__ == "__main__":
    # Example usage
    train_x, test_x, train_y, test_y = getEmbeddings('datasets/train.csv')