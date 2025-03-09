from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from typing import List, Tuple, Optional
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    user_id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)

class Word(Base):
    __tablename__ = 'words'
    word_id = Column(Integer, primary_key=True)
    word = Column(String, unique=True, nullable=False)
    language = Column(String, nullable=False)  # e.g., "Spanish"
    definition = Column(String)  # Optional definition

class UserProgress(Base):
    __tablename__ = 'user_progress'
    user_id = Column(Integer, ForeignKey('users.user_id'), primary_key=True)
    word_id = Column(Integer, ForeignKey('words.word_id'), primary_key=True)
    correct_count = Column(Integer, default=0)
    incorrect_count = Column(Integer, default=0)
    last_attempted = Column(DateTime, default=datetime.utcnow)
    user = relationship("User")
    word = relationship("Word")

class LanguageLearningDatabase:
    def __init__(self, db_name: str = "sqlite:///language_learning.db"):
        self.engine = create_engine(db_name)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def add_user(self, username: str) -> int:
        session = self.Session()
        try:
            user = User(username=username)
            session.add(user)
            session.commit()
            return user.user_id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def add_word(self, word: str, language: str, definition: Optional[str] = None) -> int:
        session = self.Session()
        try:
            word_obj = session.query(Word).filter_by(word=word, language=language).first()
            if not word_obj:
                word_obj = Word(word=word, language=language, definition=definition)
                session.add(word_obj)
                session.commit()
            return word_obj.word_id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def update_user_word(self, user_id: int, word_id: int, correct: bool) -> None:
        session = self.Session()
        try:
            progress = session.query(UserProgress).filter_by(user_id=user_id, word_id=word_id).first()
            if not progress:
                progress = UserProgress(user_id=user_id, word_id=word_id)
            if correct:
                progress.correct_count += 1
            else:
                progress.incorrect_count += 1
            progress.last_attempted = datetime.utcnow()
            session.add(progress)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_user_progress(self, user_id: int) -> List[Tuple[str, int, int]]:
        session = self.Session()
        try:
            progress = session.query(UserProgress).filter_by(user_id=user_id).all()
            return [(p.word.word, p.correct_count, p.incorrect_count) for p in progress]
        finally:
            session.close()

    def get_all_words(self, language: Optional[str] = None) -> List[str]:
        session = self.Session()
        try:
            query = session.query(Word)
            if language:
                query = query.filter_by(language=language)
            words = query.all()
            return [w.word for w in words]
        finally:
            session.close()

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        session = self.Session()
        try:
            return session.query(User).filter_by(user_id=user_id).first()
        finally:
            session.close()