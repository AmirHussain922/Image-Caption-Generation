
"""
Evaluation Module - BLEU score calculation for image captioning
"""

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np

def calculate_bleu_scores(references, hypotheses, smoothing_function=None):
    """
    Calculate BLEU-1 to BLEU-4 scores
    :param references: list of lists of reference captions
    :param hypotheses: list of generated captions
    :return: dict with BLEU scores
    """
    if smoothing_function is None:
        smoothing_function = SmoothingFunction().method4
        
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smoothing_function)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function)
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.333, 0.333, 0.333, 0), smoothing_function=smoothing_function)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)
    
    return {
        'BLEU-1': np.round(bleu1, 4),
        'BLEU-2': np.round(bleu2, 4),
        'BLEU-3': np.round(bleu3, 4),
        'BLEU-4': np.round(bleu4, 4)
    }
    
def print_evaluation_report(scores):
    """Print formatted evaluation report"""
    print("\n" + "="*50)
    print("EVALUATION REPORT")
    print("="*50)
    for metric, value in scores.items():
        print(f"  {metric}: {value*100:.2f}%")
    print("="*50 + "\n")
