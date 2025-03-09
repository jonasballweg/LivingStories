import os
from typing import Optional, List, Dict, Tuple
from openai import OpenAI
from db_operation import LanguageLearningDatabase  # Only import LanguageLearningDatabase

class InteractiveStoryGenerator:
    """Generates interactive story segments and language tasks for the user."""
    def __init__(self, api_key: Optional[str] = None, target_language: str = "Spanish"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Set it via parameter or OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=self.api_key)
        self.target_language = target_language
        self.story_context: List[Dict[str, str]] = []  # Maintains story continuity

    def generate_story_segment(self, user_response: Optional[str] = None) -> Dict[str, str]:
        """Generates a story segment with a task and target word."""
        if user_response:
            self.story_context.append({"role": "user", "content": user_response})
        prompt = (
            f"Generate a short interactive story segment in English set in a {self.target_language}-speaking context. "
            f"End with a task for the user to use a specific {self.target_language} word in a sentence. "
            f"Return the response in this format: 'Story: [story text] Task: Use the word \"[word]\" in a sentence.'"
        )
        response = self._call_llm(prompt)
        # Parse response into structured format
        try:
            story_part, task_part = response.split("Task:", 1)
            story = story_part.replace("Story:", "").strip()
            word_start = task_part.find('"') + 1
            word_end = task_part.find('"', word_start)
            word = task_part[word_start:word_end]
            task = task_part.strip()
        except ValueError:
            # Fallback if LLM response is malformed
            story = "An error occurred in the story."
            task = "Use the word 'error' in a sentence."
            word = "error"
        self.story_context.append({"role": "assistant", "content": response})
        return {"story": story, "task": task, "word": word}

    def _call_llm(self, prompt: str) -> str:
        """Calls the OpenAI API with retry logic."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You are a creative story generator."}] + 
                         self.story_context + [{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM Error: {e}")
            return "Story: An error occurred. Task: Use the word \"error\" in a sentence."

class LanguageEvaluator:
    """Evaluates user responses to language tasks and provides feedback."""
    def __init__(self, api_key: Optional[str] = None, target_language: str = "Spanish"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required.")
        self.client = OpenAI(api_key=self.api_key)
        self.target_language = target_language

    def evaluate_response(self, task: str, user_response: str) -> Tuple[bool, str]:
        """Evaluates the user's response and returns correctness with feedback."""
        prompt = (
            f"Evaluate this response in {self.target_language} for the task '{task}': '{user_response}'. "
            f"Return 'Correct: [explanation]' if correct, or 'Incorrect: [explanation]' if wrong."
        )
        feedback = self._call_llm(prompt)
        correct = feedback.startswith("Correct:")
        explanation = feedback.split(":", 1)[1].strip() if ":" in feedback else feedback
        return correct, explanation

    def _call_llm(self, prompt: str) -> str:
        """Calls the OpenAI API for evaluation."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": f"You are a {self.target_language} language expert."}, 
                          {"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.5,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM Error: {e}")
            return "Incorrect: Unable to evaluate response due to an error."

class ProgressTracker:
    """Tracks user progress and interacts with the database."""
    def __init__(self, db: LanguageLearningDatabase):
        self.db = db

    def add_user(self, username: str) -> int:
        """Adds a new user and returns their user_id."""
        return self.db.add_user(username)

    def add_word(self, word: str, language: str = "Spanish", definition: Optional[str] = None) -> int:
        """Adds a word with language context and returns its word_id."""
        return self.db.add_word(word, language, definition)

    def update_progress(self, user_id: int, word_id: int, correct: bool) -> None:
        """Updates user progress based on correctness."""
        self.db.update_user_word(user_id, word_id, correct)

    def get_user_progress(self, user_id: int) -> List[Tuple[str, int, int]]:
        """Retrieves user progress as (word, correct_count, incorrect_count)."""
        return self.db.get_user_progress(user_id)

class LearningSessionManager:
    """Manages the interactive learning session."""
    def __init__(self, user_id: int, db: LanguageLearningDatabase, target_language: str = "Spanish"):
        self.user_id = user_id
        self.story_generator = InteractiveStoryGenerator(target_language=target_language)
        self.evaluator = LanguageEvaluator(target_language=target_language)
        self.progress_tracker = ProgressTracker(db)
        self.target_language = target_language

    def start_session(self) -> None:
        """Runs an interactive learning session."""
        print(f"Welcome to your {self.target_language} learning adventure!")
        story_data = self.story_generator.generate_story_segment()
        print(f"\n{story_data['story']}\n{story_data['task']}")

        while True:
            user_response = input("Your response (or 'exit' to quit): ")
            if user_response.lower() == "exit":
                break

            correct, feedback = self.evaluator.evaluate_response(story_data["task"], user_response)
            print(f"Feedback: {feedback}")

            word_id = self.progress_tracker.add_word(story_data["word"], self.target_language)
            self.progress_tracker.update_progress(self.user_id, word_id, correct)

            progress = self.progress_tracker.get_user_progress(self.user_id)
            print("Your progress (word, correct, incorrect):", progress)

            story_data = self.story_generator.generate_story_segment(user_response)
            print(f"\n{story_data['story']}\n{story_data['task']}")