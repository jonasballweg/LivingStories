import sqlite3
from typing import Optional

class LanguageLearningDatabase:
    """Handles all database operations for the language learning app."""
    def __init__(self, db_name: str = "language_learning.db"):
        self.db_name = db_name
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()

    def create_tables(self):
        """Creates necessary tables if they don't exist."""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS words (
                word_id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT NOT NULL UNIQUE
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_progress (
                user_id INTEGER,
                word_id INTEGER,
                correct_count INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                FOREIGN KEY (word_id) REFERENCES words(word_id),
                PRIMARY KEY (user_id, word_id)
            )
        ''')
        self.conn.commit()

    def add_user(self, username: str) -> int:
        """Adds a new user and returns their user_id."""
        self.cursor.execute("INSERT INTO users (username) VALUES (?)", (username,))
        self.conn.commit()
        self.cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
        return self.cursor.fetchone()[0]

    def add_word(self, word: str) -> int:
        """Adds a new word and returns its word_id."""
        self.cursor.execute("INSERT OR IGNORE INTO words (word) VALUES (?)", (word,))
        self.conn.commit()
        self.cursor.execute("SELECT word_id FROM words WHERE word = ?", (word,))
        return self.cursor.fetchone()[0]

    def update_user_word(self, user_id: int, word_id: int, correct: bool):
        """Updates the user's progress for a specific word."""
        self.cursor.execute('''
            INSERT OR REPLACE INTO user_progress (user_id, word_id, correct_count)
            VALUES (?, ?, (SELECT COALESCE((SELECT correct_count FROM user_progress WHERE user_id = ? AND word_id = ?), 0) + ?))
        ''', (user_id, word_id, user_id, word_id, 1 if correct else 0))
        self.conn.commit()

    def close(self):
        """Closes the database connection."""
        self.conn.close()

class ProgressTracker:
    """Manages user progress tracking and database updates."""
    def __init__(self, db_manager: LanguageLearningDatabase):
        self.db_manager = db_manager

    def update_progress(self, user_id: int, word_id: int, correct: bool):
        """Updates the user's progress in the database."""
        self.db_manager.update_user_word(user_id, word_id, correct)

    def add_user(self, username: str) -> int:
        """Adds a user via the database manager."""
        return self.db_manager.add_user(username)

    def add_word(self, word: str) -> int:
        """Adds a word via the database manager."""
        return self.db_manager.add_word(word)