import openai
from typing import List, Dict

# Example structure for userVocabulary: 
# [
#   {
#       "word": "hello",
#       "score": 1.0,
#       "synonyms": ["hi", "greetings"]
#   },
#   ...
# ]

def generateProficiencyAlignedText(
    generationPrompt: str,
    proficiencyLevel: int,
    userVocabulary: List[Dict]
) -> str:
    """
    Generate text from an LLM guided by the user's proficiency level.
    Then evaluate the text, and replace words that the user likely does not know
    with more familiar words or simpler synonyms. 
    """
    
    # Step 1: Generate initial text using LLM (pseudo-code/OpenAI as example)
    # You must configure your openai.api_key or other credentials.
    # Provide instructions to the LLM about the desired difficulty level.
    system_prompt = f"""
    You are a Chinese language tutor. 
    Write a story in Mandarin suitable for an HSK {proficiencyLevel} learner. 
    Keep vocabulary and grammar level appropriate to HSK {proficiencyLevel}.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or your chosen model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": generationPrompt},
            ],
            temperature=0.7
        )
        generated_text = response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return ""

    # Step 2: Evaluate words in the generated text
    # Split on whitespace (for simplicity) - real approach should handle punctuation, etc.
    generated_words = generated_text.split()
    
    # Identify words the user may not know (score < 0.5 as an example threshold)
    # In practice, you may want to use word embeddings or synonyms 
    # relevant to context, but here we simply look them up in userVocabulary.
    user_vocab_map = {entry["word"]: entry for entry in userVocabulary}
    unknown_words = []
    for word in generated_words:
        # Simplify check by ignoring punctuation/case (depends on your approach)
        cleaned_word = word.strip(".,!?，。？").lower()
        
        if cleaned_word in user_vocab_map:
            if user_vocab_map[cleaned_word]["score"] < 0.5:
                unknown_words.append(cleaned_word)
        else:
            # Word not found in user vocabulary is considered unknown
            unknown_words.append(cleaned_word)
    
    # Step 3: Replace unknown words with synonyms that the user may know better
    # This is a simplistic approach: we pick the first known synonym
    # Real approach might use vector similarity or context checks (e.g., sentence embeddings).
    adjusted_words = []
    for word in generated_words:
        cleaned_word = word.strip(".,!?，。？").lower()
        punctuation_prefix = word[0] if not word[0].isalnum() else ""
        punctuation_suffix = word[-1] if not word[-1].isalnum() else ""
        
        if cleaned_word in unknown_words:
            vocab_entry = user_vocab_map.get(cleaned_word, None)
            if vocab_entry and "synonyms" in vocab_entry:
                # Find a synonym the user knows well
                chosen_synonym = next(
                    (syn for syn in vocab_entry["synonyms"] 
                     if syn in user_vocab_map 
                     and user_vocab_map[syn]["score"] >= 0.5),
                    cleaned_word  # fallback if no good synonym is found
                )
                adjusted_words.append(f"{punctuation_prefix}{chosen_synonym}{punctuation_suffix}")
            else:
                # If no synonyms available, keep the original word or replace with placeholder
                adjusted_words.append(f"{punctuation_prefix}{cleaned_word}{punctuation_suffix}")
        else:
            adjusted_words.append(word)
    
    final_text = " ".join(adjusted_words)
    return final_text

if __name__ == "__main__":
    # EXAMPLE USAGE (replace with real data and real API keys):
    prompt_example = "Write a brief story about going to the park in Mandarin."
    user_vocab_example = [
        {"word": "我", "score": 1.0, "synonyms": []},
        {"word": "公园", "score": 0.8, "synonyms": []},
        {"word": "美丽", "score": 0.4, "synonyms": ["漂亮", "好看"]},
        # ...
    ]
    
    proficiency_level_example = 3
    
    story = generateProficiencyAlignedText(
        generationPrompt=prompt_example,
        proficiencyLevel=proficiency_level_example,
        userVocabulary=user_vocab_example
    )
    print("Final Adjusted Story:\n", story)
