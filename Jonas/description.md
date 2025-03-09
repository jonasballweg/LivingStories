
# Function Overview
Function: adjustTextToProficiency(text, proficiencyLevel, userVocabulary)
Adjusting an Existing Text for the User’s Language Proficiency
Parameters:
- text that should be changed
- proficiencyLevel: The user’s language proficiency (e.g., HSK 1-9 for Mandarin, A1-C2 for European languages).
- userVocabulary: A list of all words which the user knows. Every word gets a score assigned between 0 (user does not know the word at all) to 1 (user knows the word perfectly).

# Code Implementation

1. **Initial Text Generation and Setup**  
   Begin by producing or obtaining the source text that needs to be adjusted. This text should be generated or reviewed with the target language proficiency level in mind (for example, A1 through C2 for European languages or HSK levels for Mandarin). At this stage, establish the user’s vocabulary list along with their familiarity scores for each word. This list is crucial for later comparisons.

2. **Identification and Preservation of Proper Nouns and Domain-Specific Terms**  
   As you process the text, first mark proper nouns, place names, culturally significant terms, and technical vocabulary that belongs to specialized domains (such as medical or legal jargon). These terms should remain unchanged unless a modification is absolutely necessary. This ensures that the integrity of the content, context, and domain-specific meanings is maintained.

3. **Sentence-Level Evaluation and Contextual Analysis**  
   Analyze the text sentence by sentence to evaluate if any words exceed the user’s vocabulary level. For each sentence, compare each word against the user’s vocabulary list by using embeddings. Words with low familiarity scores are flagged. Before comparing the words, a lemmatizer might be necessary.

4. **Vocabulary Comparison and Synonym Replacement**  
   - For each flagged word, search for simpler alternatives or synonyms that the user is likely to know. Instead of relying solely on isolated word embeddings, consider using sentence embeddings to find context-appropriate replacements. This helps ensure that the substituted words naturally fit the overall meaning of the sentence.
        - Ensure that the replacements do not inadvertently change the sentence’s intended meaning or disrupt the overall narrative flow.
        - Look not only at individual sentences but also at how they connect with the sentences before and after. This is important to maintain overall coherence and context.
        - Preserve the grammatical structure by ensuring that any changes to vocabulary do not disrupt verb conjugations, noun-adjective agreements, or idiomatic expressions.

5. **Progressive Simplification Strategy**  
   If a sentence contains multiple words outside the user’s known vocabulary, consider:
   - First, replacing individual words while preserving the original sentence structure.
   - If too many replacements are necessary and the sentence becomes awkward or overly complex, rewrite or simplify the entire sentence.
   - If simplifying a single sentence is insufficient to maintain clarity due to severe vocabulary limitations, then simplify the entire paragraph. The goal is to keep the content understandable without losing essential details or context.

6. **Maintaining Grammar, Syntax, and Idiomatic Integrity**  
   During the substitution process:
   - Verify that any new words fit the sentence’s grammatical structure. For example, check that verb forms remain correct and that adjectives agree with the nouns they modify.
   - For idiomatic expressions that do not have straightforward simpler alternatives, consider rewriting the expression into a more direct or literal form while still conveying the original intent.

8. **Final Review and Quality Check**  
   Once the entire text has been processed:
   - Conduct a holistic review to ensure that all replacements and simplifications maintain coherence, readability, and context integrity across the document.
   - Double-check that proper nouns, technical terms, and culturally significant elements remain intact and that the overall message is preserved.
   - 

If necesary, use frameworks like LangChain or AutoGen (or any more up-to-date frameworks) in the code.  
