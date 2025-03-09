import os
import re
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download necessary NLTK resources
def download_nltk_resources():
    """Download required NLTK resources if they're not already available."""
    resources = ['punkt', 'averaged_perceptron_tagger', 'wordnet', 'stopwords']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' 
                           else f'taggers/{resource}' if resource == 'averaged_perceptron_tagger'
                           else f'corpora/{resource}')
        except LookupError:
            nltk.download(resource)

# Initialize resources
download_nltk_resources()

# Initialize OpenAI client
def get_openai_client():
    """Initialize and return the OpenAI client."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found in environment variables")
    return OpenAI(api_key=api_key)

client = get_openai_client()

@lru_cache(maxsize=1024)
def get_embedding(text: str) -> List[float]:
    """
    Get embedding for text using OpenAI's API with caching for efficiency.
    
    Args:
        text: The text to get embedding for
        
    Returns:
        A list of floats representing the embedding vector
    """
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding for text: {e}")
        # Return a zero vector as fallback (size for ada-002 model)
        return [0.0] * 1536

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity value between -1 and 1
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def is_proper_noun_or_technical_term(word: str, pos: str) -> bool:
    """
    Check if a word is a proper noun or likely a technical/domain-specific term.
    
    Args:
        word: The word to check
        pos: Part-of-speech tag for the word
        
    Returns:
        True if the word is a proper noun or technical term, False otherwise
    """
    # Check if word is a proper noun based on POS tag
    if pos.startswith('NNP'):
        return True
    
    # Additional checks could be added here for domain-specific terms
    # This could involve checking against specialized dictionaries
    
    return False

def identify_words_to_replace(sentence: str, user_vocabulary: Dict[str, float], 
                             threshold: float = 0.5) -> List[Tuple[str, str]]:
    """
    Identify words in a sentence that need to be replaced based on user vocabulary.
    
    Args:
        sentence: The sentence to analyze
        user_vocabulary: Dictionary mapping words to familiarity scores (0-1)
        threshold: Minimum familiarity score for words to be considered known
        
    Returns:
        List of (word, pos) tuples for words that need replacement
    """
    words = word_tokenize(sentence)
    tagged_words = pos_tag(words)
    words_to_replace = []
    
    lemmatizer = WordNetLemmatizer()
    
    for word, pos in tagged_words:
        # Skip punctuation, numbers, stopwords, and proper nouns
        if (not word.isalpha() or 
            word.lower() in stopwords.words('english') or 
            is_proper_noun_or_technical_term(word, pos)):
            continue
            
        # Lemmatize the word for vocabulary lookup
        word_lemma = lemmatizer.lemmatize(word.lower())
        
        # Check if the word is in user vocabulary with sufficient familiarity
        if word_lemma not in user_vocabulary or user_vocabulary[word_lemma] < threshold:
            words_to_replace.append((word, pos))
    
    return words_to_replace

def find_best_synonym(word: str, context: str, user_vocabulary: Dict[str, float]) -> str:
    """
    Find the best synonym for a word that matches the user's vocabulary.
    
    Args:
        word: The word to find a synonym for
        context: The sentence containing the word
        user_vocabulary: Dictionary of words the user knows with familiarity scores
        
    Returns:
        The best synonym found or original word if no good synonym available
    """
    # Get embedding for the word in context
    target_embedding = get_embedding(f"{context} {word}")
    
    # Find known words in user vocabulary (score > 0.6)
    known_words = {w: score for w, score in user_vocabulary.items() if score > 0.6}
    
    if not known_words:
        return word
    
    # Calculate similarity scores for each known word
    candidates = []
    for candidate, score in known_words.items():
        candidate_in_context = context.replace(word, candidate)
        candidate_embedding = get_embedding(candidate_in_context)
        similarity = cosine_similarity(target_embedding, candidate_embedding)
        candidates.append((candidate, similarity, score))
    
    # Sort by similarity and user familiarity
    candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    
    # Return the best candidate if it's good enough
    if candidates and candidates[0][1] > 0.7:
        return candidates[0][0]
    
    # If no good candidates, use LLM to find alternative
    return find_replacement_using_llm(word, context, user_vocabulary)

def find_replacement_using_llm(word: str, context: str, user_vocabulary: Dict[str, float]) -> str:
    """
    Use LLM to find a suitable replacement for a word based on context and user vocabulary.
    
    Args:
        word: The word to replace
        context: The sentence containing the word
        user_vocabulary: Dictionary of words the user knows with familiarity scores
        
    Returns:
        A suitable replacement word or phrase
    """
    # Extract sample of known words for context
    known_words = [w for w, score in user_vocabulary.items() if score > 0.7]
    sample_size = min(30, len(known_words))
    known_sample = ", ".join(known_words[:sample_size])
    
    prompt = f"""
    I need to simplify this sentence for a language learner:
    
    "{context}"
    
    I need a simpler alternative for the word "{word}" that preserves the meaning in this context.
    The learner knows these words: {known_sample}
    
    Provide only the alternative word or short phrase, nothing else.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specializing in language simplification."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50
        )
        
        replacement = response.choices[0].message.content.strip()
        
        # If response is too long, take just the first few words
        if len(replacement.split()) > 3:
            replacement = " ".join(replacement.split()[:3])
            
        return replacement
    except Exception as e:
        logger.error(f"Error finding replacement for '{word}': {e}")
        return word  # Return original word as fallback

def simplify_sentence(sentence: str, proficiency_level: str, user_vocabulary: Dict[str, float]) -> str:
    """
    Simplify a sentence by replacing complex words or rewriting if necessary.
    
    Args:
        sentence: The sentence to simplify
        proficiency_level: The target language proficiency level
        user_vocabulary: Dictionary of words the user knows with familiarity scores
        
    Returns:
        The simplified sentence
    """
    words_to_replace = identify_words_to_replace(sentence, user_vocabulary)
    
    # If no words need replacement, return the original sentence
    if not words_to_replace:
        return sentence
    
    # If too many words need replacement (>30%), rewrite the entire sentence
    if len(words_to_replace) / len(word_tokenize(sentence)) > 0.3:
        return rewrite_sentence(sentence, proficiency_level, user_vocabulary)
    
    # Replace individual words
    modified_sentence = sentence
    for word, _ in words_to_replace:
        replacement = find_best_synonym(word, sentence, user_vocabulary)
        
        # Preserve capitalization
        if word[0].isupper() and len(replacement) > 0:
            replacement = replacement[0].upper() + replacement[1:]
            
        # Replace the word in the sentence (with word boundary check)
        modified_sentence = re.sub(r'\b' + re.escape(word) + r'\b', replacement, modified_sentence)
    
    return modified_sentence

def rewrite_sentence(sentence: str, proficiency_level: str, user_vocabulary: Dict[str, float]) -> str:
    """
    Completely rewrite a sentence to match the target proficiency level.
    
    Args:
        sentence: The sentence to rewrite
        proficiency_level: The target language proficiency level
        user_vocabulary: Dictionary of words the user knows with familiarity scores
        
    Returns:
        The rewritten sentence
    """
    # Extract sample of known words for context
    known_words = [w for w, score in user_vocabulary.items() if score > 0.7]
    sample_size = min(30, len(known_words))
    known_sample = ", ".join(known_words[:sample_size])
    
    prompt = f"""
    Please simplify this sentence for a language learner at {proficiency_level} level:
    
    "{sentence}"
    
    The learner knows these words: {known_sample}
    
    Rewrite the sentence using simple words that maintain the original meaning.
    Use only words from their known vocabulary when possible.
    Provide only the rewritten sentence, nothing else.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specializing in language simplification."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error rewriting sentence: {e}")
        return sentence  # Return original sentence as fallback

def ensure_paragraph_coherence(paragraph: str, proficiency_level: str, user_vocabulary: Dict[str, float]) -> str:
    """
    Ensure a paragraph maintains coherence after sentence-level simplifications.
    
    Args:
        paragraph: The paragraph with simplified sentences
        proficiency_level: The target language proficiency level
        user_vocabulary: Dictionary of words the user knows with familiarity scores
        
    Returns:
        The paragraph with improved coherence
    """
    # Extract sample of known words for context
    known_words = [w for w, score in user_vocabulary.items() if score > 0.7]
    sample_size = min(30, len(known_words))
    known_sample = ", ".join(known_words[:sample_size])
    
    prompt = f"""
    I have simplified this paragraph for a language learner at {proficiency_level} level:
    
    "{paragraph}"
    
    The learner knows these words: {known_sample}
    
    Please review this paragraph and make any necessary adjustments to:
    1. Improve coherence and natural flow
    2. Maintain simplicity at {proficiency_level} level
    3. Preserve the original meaning
    
    Provide only the revised paragraph, nothing else.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specializing in language education."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error ensuring paragraph coherence: {e}")
        return paragraph  # Return original paragraph as fallback

def final_quality_check(text: str, proficiency_level: str, user_vocabulary: Dict[str, float]) -> str:
    """
    Perform a final quality check on the adjusted text.
    
    Args:
        text: The adjusted text to check
        proficiency_level: The target language proficiency level
        user_vocabulary: Dictionary of words the user knows with familiarity scores
        
    Returns:
        The final adjusted text after quality check
    """
    prompt = f"""
    I have adjusted this text for a language learner at {proficiency_level} level:
    
    "{text}"
    
    Please review this text and make any necessary adjustments to:
    1. Ensure it consistently matches the {proficiency_level} proficiency level
    2. Maintain coherence and natural flow
    3. Preserve the original meaning and essential information
    4. Keep proper nouns and technical terms intact where appropriate
    
    Provide only the final adjusted text, nothing else.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specializing in language education and text adaptation."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error in final quality check: {e}")
        return text  # Return pre-check version as fallback

def adjustTextToProficiency(text: str, proficiency_level: str, user_vocabulary: Dict[str, float]) -> str:
    """
    Adjust a text to match the user's language proficiency level.
    
    This function takes a text, analyzes it against the user's known vocabulary and proficiency level,
    and then simplifies it appropriately while preserving the original meaning. It handles:
    1. Identification and preservation of proper nouns and domain-specific terms
    2. Sentence-level evaluation and contextual analysis
    3. Vocabulary comparison and synonym replacement
    4. Progressive simplification strategy
    5. Maintenance of grammar, syntax, and idiomatic integrity
    6. Final review and quality check
    
    Args:
        text: The original text to be adjusted
        proficiency_level: The user's language proficiency level (e.g., "HSK 3" for Mandarin, "B1" for European languages)
        user_vocabulary: Dictionary mapping words to familiarity scores (0-1)
        
    Returns:
        The adjusted text matching the user's proficiency level
    """
    logger.info(f"Starting text adjustment for proficiency level: {proficiency_level}")
    
    # Process the text by paragraphs
    paragraphs = text.split('\n')
    adjusted_paragraphs = []
    
    for i, paragraph in enumerate(paragraphs):
        if not paragraph.strip():  # Skip empty paragraphs
            adjusted_paragraphs.append("")
            continue
            
        logger.info(f"Processing paragraph {i+1}/{len(paragraphs)}")
        
        # Process each sentence in the paragraph
        sentences = sent_tokenize(paragraph)
        simplified_sentences = []
        
        for j, sentence in enumerate(sentences):
            logger.info(f"Simplifying sentence {j+1}/{len(sentences)} in paragraph {i+1}")
            simplified = simplify_sentence(sentence, proficiency_level, user_vocabulary)
            simplified_sentences.append(simplified)
        
        # Join sentences back into a paragraph
        simplified_paragraph = " ".join(simplified_sentences)
        
        # Check if paragraph needs coherence improvement
        # (if more than 40% of sentences were modified)
        modified_count = sum(1 for orig, simp in zip(sentences, simplified_sentences) if orig != simp)
        if modified_count / len(sentences) > 0.4:
            logger.info(f"Improving coherence for heavily modified paragraph {i+1}")
            simplified_paragraph = ensure_paragraph_coherence(
                simplified_paragraph, proficiency_level, user_vocabulary
            )
        
        adjusted_paragraphs.append(simplified_paragraph)
    
    # Join paragraphs back into text
    adjusted_text = '\n'.join(adjusted_paragraphs)
    
    # Final quality check
    logger.info("Performing final quality check")
    final_text = final_quality_check(adjusted_text, proficiency_level, user_vocabulary)
    
    logger.info("Text adjustment completed")
    return final_text

# Example usage
if __name__ == "__main__":
    # Sample vocabulary (in a real scenario, this would be loaded from a database)
    sample_vocabulary = {
        "hello": 1.0,
        "good": 0.9,
        "morning": 0.8,
        "name": 0.95,
        "is": 1.0,
        "what": 0.9,
        "your": 0.9,
        "like": 0.8,
        "food": 0.9,
        "eat": 0.85,
        "day": 0.9,
        "today": 0.85,
        "my": 1.0,
        "i": 1.0,
        "you": 1.0,
        "will": 0.9,
        "guide": 0.7,
        "we": 0.95,
        "city": 0.8,
        "old": 0.9,
        "see": 0.95,
        "look": 0.9,
        "want": 0.9,
        "start": 0.85,
        "building": 0.8,
        "church": 0.75,
        # Add more vocabulary as needed
    }
    
    sample_text = """
    Good morning! My name is Sarah, and I'll be your guide today. 
    We'll be exploring the magnificent ancient architecture of this historic city.
    The intricate facades and ornate decorations demonstrate the remarkable craftsmanship of medieval artisans.
    Would you like to commence our tour at the cathedral or the municipal building?
    """
    
    proficiency_level = "A2"  # CEFR level for European languages
    
    adjusted_text = adjustTextToProficiency(sample_text, proficiency_level, sample_vocabulary)
    print("Original text:")
    print(sample_text)
    print("\nAdjusted text:")
    print(adjusted_text)
