from db_operation import LanguageLearningDatabase
from llm_agent import LearningSessionManager

if __name__ == "__main__":
    api_key = "your-openai-api-key"  # Replace with your actual key
    db = LanguageLearningDatabase()
    user_id = db.add_user("Alice")  # Add user directly or via ProgressTracker
    session_manager = LearningSessionManager(user_id=user_id, db=db, target_language="Spanish")
    session_manager.start_session()