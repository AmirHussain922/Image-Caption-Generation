
"""
Text Preprocessing and Tokenization Module
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import re

class TextProcessor:
    def __init__(self, vocab_size=10000, oov_token="<OOV>"):
        self.tokenizer = None
        self.vocab_size = vocab_size
        self.oov_token = oov_token
        self.max_sequence_length = 0
        
    def clean_text(self, text):
        """Clean caption text"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = text.strip()
        return f"<start> {text} <end>"
        
    def fit(self, captions):
        """Fit tokenizer on training captions"""
        cleaned_captions = [self.clean_text(c) for c in captions]
        
        self.tokenizer = Tokenizer(
            num_words=self.vocab_size,
            oov_token=self.oov_token,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        )
        self.tokenizer.fit_on_texts(cleaned_captions)
        
        self.max_sequence_length = max(
            len(seq) for seq in self.tokenizer.texts_to_sequences(cleaned_captions)
        )
        
    def texts_to_sequences(self, texts):
        """Convert text to sequences"""
        cleaned_texts = [self.clean_text(t) for t in texts]
        return self.tokenizer.texts_to_sequences(cleaned_texts)
    
    def sequences_to_texts(self, sequences):
        """Convert sequences back to text"""
        return self.tokenizer.sequences_to_texts(sequences)
        
    def pad_sequences(self, sequences):
        """Pad sequences to uniform length"""
        return pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')
        
    def index_to_word(self, idx):
        if self.tokenizer and idx in self.tokenizer.index_word:
            return self.tokenizer.index_word[idx]
        return self.oov_token
        
    def word_to_index(self, word):
        if self.tokenizer and word in self.tokenizer.word_index:
            return self.tokenizer.word_index[word]
        return self.tokenizer.word_index[self.oov_token]
        
    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump({
                'tokenizer': self.tokenizer,
                'vocab_size': self.vocab_size,
                'max_sequence_length': self.max_sequence_length
            }, f)
            
    @classmethod
    def load(cls, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        processor = cls(vocab_size=data['vocab_size'])
        processor.tokenizer = data['tokenizer']
        processor.max_sequence_length = data['max_sequence_length']
        return processor
