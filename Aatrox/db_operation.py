import sqlite3
import datetime

class DatabaseManager:
    def __init__(self, db_name="language_learning.db"):
        self.db_name = db_name

    def connect_db(self):
        """Connects to the SQLite database and returns the connection and cursor."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        return conn, cursor

    def create_tables(self):
        """Creates tables for users, words, and user_words (tracking user progress)."""
        conn, cursor = self.connect_db()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT
        )''')

        cursor.execute('''CREATE TABLE IF NOT EXISTS words (
            id INTEGER PRIMARY KEY,
            word TEXT,
            definition TEXT,
            difficulty INTEGER
        )''')

        cursor.execute('''CREATE TABLE IF NOT EXISTS user_words (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            word_id INTEGER,
            mastery_score REAL,
            last_reviewed TIMESTAMP,
            next_review TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(word_id) REFERENCES words(id)
        )''')
        conn.commit()
        conn.close()

    def add_user(self, name):
        """Adds a new user to the database."""
        conn, cursor = self.connect_db()
        cursor.execute('INSERT INTO users (name) VALUES (?)', (name,))
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return user_id

    def add_word(self, word, definition, difficulty):
        """Adds a new word to the database."""
        conn, cursor = self.connect_db()
        cursor.execute('INSERT INTO words (word, definition, difficulty) VALUES (?, ?, ?)',
                       (word, definition, difficulty))
        conn.commit()
        word_id = cursor.lastrowid
        conn.close()
        return word_id

    def initialize_user_word(self, user_id, word_id):
        """Initializes a user-word record with a default mastery score."""
        conn, cursor = self.connect_db()
        now = datetime.datetime.now()
        cursor.execute('''INSERT INTO user_words (user_id, word_id, mastery_score, last_reviewed, next_review)
                          VALUES (?, ?, ?, ?, ?)''', (user_id, word_id, 0.0, now, now))
        conn.commit()
        conn.close()

    def update_user_word(self, user_id, word_id, correct):
        """Updates the user's mastery score for a word based on correctness."""
        conn, cursor = self.connect_db()
        cursor.execute('SELECT mastery_score FROM user_words WHERE user_id=? AND word_id=?', (user_id, word_id))
        result = cursor.fetchone()

        if result:
            mastery_score = result[0]
        else:
            print("User-word record not found. Initializing.")
            self.initialize_user_word(user_id, word_id)
            mastery_score = 0.0

        mastery_score = mastery_score + 1 if correct else max(mastery_score - 1, 0)
        now = datetime.datetime.now()
        interval_days = 1 + mastery_score
        next_review = now + datetime.timedelta(days=interval_days)

        cursor.execute('''UPDATE user_words
                          SET mastery_score=?, last_reviewed=?, next_review=?
                          WHERE user_id=? AND word_id=?''', 
                       (mastery_score, now, next_review, user_id, word_id))
        conn.commit()
        conn.close()

    def get_user_progress(self):
        """Fetches and returns all user progress records."""
        conn, cursor = self.connect_db()
        cursor.execute('SELECT * FROM user_words')
        rows = cursor.fetchall()
        conn.close()
        return rows