
"""
Inference Module for Image Caption Generator
"""

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import pickle
import os

from model import ImageCaptioningModel
from text_processor import TextProcessor

class ImageCaptioner:
    def __init__(self, tokenizer_path, model_weights_path, features_path=None):
        self.text_processor = TextProcessor.load(tokenizer_path)
        
        self.caption_model = ImageCaptioningModel(
            vocab_size=self.text_processor.vocab_size,
            max_seq_len=self.text_processor.max_sequence_length
        )
        self.caption_model.load_weights(model_weights_path)
        
        # Load VGG16 for feature extraction
        self.vgg_model = VGG16(weights='imagenet')
        self.vgg_model = tf.keras.models.Model(
            inputs=self.vgg_model.input,
            outputs=self.vgg_model.layers[-2].output
        )
        
        self.features = None
        if features_path and os.path.exists(features_path):
            with open(features_path, 'rb') as f:
                self.features = pickle.load(f)
                
    def extract_image_features(self, img_path):
        """Extract VGG16 features from single image"""
        img = load_img(img_path, target_size=(224, 224))
        img = img_to_array(img)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = preprocess_input(img)
        
        features = self.vgg_model.predict(img, verbose=0)
        return features[0]
        
    def generate_caption(self, img_path, max_len=None):
        """Generate caption for input image"""
        if max_len is None:
            max_len = self.text_processor.max_sequence_length
            
        img_features = self.extract_image_features(img_path).reshape(1, -1)
        
        # Initialize sequence with <start>
        start_idx = self.text_processor.word_to_index("<start>")
        end_idx = self.text_processor.word_to_index("<end>")
        sequence = [start_idx]
        
        for _ in range(max_len):
            seq_pad = self.text_processor.pad_sequences([sequence])
            y_pred = self.caption_model.model.predict(
                [img_features, seq_pad], verbose=0
            )
            y_pred = np.argmax(y_pred[0])
            
            if y_pred == end_idx:
                break
                
            sequence.append(y_pred)
            
        # Convert to text
        sequence_text = self.text_processor.sequences_to_texts([sequence])[0]
        sequence_text = sequence_text.replace("<start>", "").replace("<end>", "").strip()
        
        return sequence_text
        
    def batch_generate(self, img_paths):
        captions = []
        for path in img_paths:
            try:
                caption = self.generate_caption(path)
                captions.append(caption)
            except Exception as e:
                print(f"Error generating caption for {path}: {e}")
                captions.append("")
        return captions
