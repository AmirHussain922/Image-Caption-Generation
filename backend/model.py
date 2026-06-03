
"""
Image Caption Generator Model Architecture
CNN-Encoder + LSTM-Decoder with Attention
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Embedding, 
    Dropout, add, Concatenate, Layer
)
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os

class ImageCaptioningModel:
    def __init__(self, vocab_size=8821, max_seq_len=34, embedding_dim=256):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.model = None
        
    def build_image_encoder(self):
        """Build CNN encoder using VGG16 (without top classification layer)"""
        vgg = VGG16(weights='imagenet')
        vgg = Model(inputs=vgg.input, outputs=vgg.layers[-2].output)
        
        image_input = Input(shape=(4096,))
        img_features = Dropout(0.5)(image_input)
        img_features = Dense(256, activation='relu')(img_features)
        return image_input, img_features
    
    def build_text_decoder(self, img_features):
        """Build LSTM decoder for text generation"""
        text_input = Input(shape=(self.max_seq_len,))
        embedding = Embedding(self.vocab_size, self.embedding_dim, mask_zero=True)(text_input)
        embedding = Dropout(0.5)(embedding)
        
        decoder_lstm = LSTM(256, return_sequences=True)(embedding)
        decoder_lstm = Dropout(0.5)(decoder_lstm)
        decoder_lstm = LSTM(256)(decoder_lstm)
        
        decoder_concat = add([img_features, decoder_lstm])
        decoder_dense = Dense(256, activation='relu')(decoder_concat)
        outputs = Dense(self.vocab_size, activation='softmax')(decoder_dense)
        
        return text_input, outputs
    
    def build_model(self):
        """Build complete encoder-decoder model"""
        image_input, img_features = self.build_image_encoder()
        text_input, outputs = self.build_text_decoder(img_features)
        
        self.model = Model(inputs=[image_input, text_input], outputs=outputs)
        return self.model
    
    def load_weights(self, weight_path):
        if self.model is None:
            self.build_model()
        self.model.load_weights(weight_path)
        return self.model
    
    def save_weights(self, weight_path):
        if self.model:
            self.model.save_weights(weight_path)
        else:
            raise ValueError("Model not built yet")
