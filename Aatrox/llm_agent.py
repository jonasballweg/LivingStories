import os
from typing import Optional, List, Tuple
from openai import OpenAI
from db_operation import ProgressTracker

class InteractiveStoryGenerator:
    """Generates interactive story segments and tasks for the user."""
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required.")
        self.client = OpenAI(api_key=self.api_key)
        self.story_context: List[dict] = []

    def generate_story_segment(self, user_response: Optional[str] = None) -> str:
        """Generates a new story segment based on the user's response."""
        if user_response:
            self.story_context.append({"role": "user", "content": user_response})
        prompt = "Continue the story based on the previous context and give a task to the user (e.g., 'Use the word \"agua\" in a sentence')."
        response = self._call_llm(prompt)
        self.story_context.append({"role": "assistant", "content": response})
        return response

    def _call_llm(self, prompt: str) -> str:
        """Calls the OpenAI API to generate text."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You are a story generator."}] + self.story_context + [{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM Error: {e}")
            return "Error: Unable to generate story segment."

class LanguageEvaluator:
    """Evaluates user's language responses and provides feedback."""
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required.")
        self.client = OpenAI(api_key=self.api_key)

    def evaluate_response(self, task: str, user_response: str) -> str:
        """Evaluates the user's response for correctness."""
        prompt = f"Evaluate the following response for the task '{task}': '{user_response}'. Provide feedback on correctness and suggest improvements if necessary."
        return self._call_llm(prompt)

    def _call_llm(self, prompt: str) -> str:
        """Calls the OpenAI API to evaluate text."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM Error: {e}")
            return "Error: Unable to evaluate response."

class LearningSessionManager:
    """Manages the flow of learning sessions, coordinating user interactions and agent responses."""
    def __init__(self, story_generator: InteractiveStoryGenerator, language_evaluator: LanguageEvaluator, progress_tracker: ProgressTracker):
        self.story_generator = story_generator
        self.language_evaluator = language_evaluator
        self.progress_tracker = progress_tracker

    def _extract_task_and_word(self, story_segment: str) -> Tuple[str, str]:
        """Extracts the task and target word from the story segment (simplified)."""
        if "Use the word" in story_segment:
            start = story_segment.find('"') + 1
            end = story_segment.find('"', start)
            word = story_segment[start:end]
            task = f"Use the word \"{word}\" in a sentence"
            return task, word
        return "Respond to the story", "unknown"

    def start_session(self, user_id: int):
        """Starts a learning session for the user."""
        print("Welcome to the language learning adventure!")
        story_segment = self.story_generator.generate_story_segment()
        print(story_segment)

        while True:
            user_response = input("Your response: ")
            if user_response.lower() == "exit":
                break

            task, word = self._extract_task_and_word(story_segment)
            feedback = self.language_evaluator.evaluate_response(task, user_response)
            print(f"Feedback: {feedback}")

            correct = "correct" in feedback.lower()
            word_id = self.progress_tracker.add_word(word)
            self.progress_tracker.update_progress(user_id, word_id, correct)

            story_segment = self.story_generator.generate_story_segment(user_response)
            print(story_segment)