from db_operation import LanguageLearningDatabase, ProgressTracker
from llm_agent import InteractiveStoryGenerator, LanguageEvaluator, LearningSessionManager

if __name__ == "__main__":
    api_key = "your-openai-api-key"  # Replace with your actual OpenAI API key
    db_manager = LanguageLearningDatabase()
    db_manager.create_tables()

    story_generator = InteractiveStoryGenerator(api_key=api_key)
    language_evaluator = LanguageEvaluator(api_key=api_key)
    progress_tracker = ProgressTracker(db_manager)
    session_manager = LearningSessionManager(story_generator, language_evaluator, progress_tracker)

    user_id = progress_tracker.add_user("Alice")
    session_manager.start_session(user_id)

    db_manager.close()