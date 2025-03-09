import spacy
from spacy.tokens import Doc
from spacy.vocab import Vocab
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def adjustTextToProficiency(text: str, proficiencyLevel: str, userVocabulary: Dict[str, float]) -> str:
    """
    Adjusts the given text to match the user's language proficiency level.

    Parameters:
    - text: The text to be adjusted.
    - proficiencyLevel: The user's language proficiency level (e.g., HSK 1-9 for Mandarin, A1-C2 for European languages).
    - userVocabulary: A dictionary where keys are words and values are familiarity scores (0 to 1).

    Returns:
    - Adjusted text.
    """
    doc = nlp(text)
    adjusted_text = []

    for sent in doc.sents:
        adjusted_sent = []
        for token in sent:
            if token.text in userVocabulary and userVocabulary[token.text] >= 0.5:
                adjusted_sent.append(token.text)
            else:
                # Find a simpler synonym
                synonym = find_simpler_synonym(token.text, userVocabulary)
                adjusted_sent.append(synonym)
        adjusted_text.append(" ".join(adjusted_sent))

    return " ".join(adjusted_text)

def find_simpler_synonym(word: str, userVocabulary: Dict[str, float]) -> str:
    """
    Finds a simpler synonym for the given word based on the user's vocabulary.

    Parameters:
    - word: The word to find a synonym for.
    - userVocabulary: A dictionary where keys are words and values are familiarity scores (0 to 1).

    Returns:
    - A simpler synonym if found, otherwise the original word.
    """
    word_vector = nlp(word).vector
    max_similarity = -1
    best_synonym = word

    for vocab_word, score in userVocabulary.items():
        if score >= 0.5:
            vocab_word_vector = nlp(vocab_word).vector
            similarity = cosine_similarity([word_vector], [vocab_word_vector])[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                best_synonym = vocab_word

    return best_synonym
