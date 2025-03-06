import os

from datetime import datetime
from sqlalchemy import Column, Integer, String, LargeBinary, BigInteger, Boolean, DateTime, ForeignKey
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import mapped_column, sessionmaker, declarative_base, relationship
from dotenv import load_dotenv

# Настройка подключения к БД
load_dotenv()
engine =  create_async_engine(url=os.getenv('SQLALCHEMY_URL'))
enginephoto = create_async_engine(url=os.getenv('SQLITE_PHOTO'))
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
async_session_photo = sessionmaker(enginephoto, class_=AsyncSession, expire_on_commit=False)

BaseFace = declarative_base()
BasePhoto = declarative_base()

# отдельные таблицы для пользователей и их эмбеддингов
# Пользователи
class User(BaseFace):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    tg_id = Column(BigInteger, nullable=False)  # Убрали unique=True
    # tg_id = Column(BigInteger, unique=True, nullable=False)  # ID в Telegram уникальный
    name = Column(String(100), nullable=False)               # Основное имя
    created_at = Column(DateTime, default=datetime.now)
    embeddings = relationship("FaceEmbedding", back_populates="user", cascade="all, delete-orphan")


# Эмбеддинги
class FaceEmbedding(BaseFace):
    __tablename__ = "embeddings"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    embedding = Column(LargeBinary, nullable=False)          # Бинарные данные
    created_at = Column(DateTime, default=datetime.now)
    user = relationship("User", back_populates="embeddings")

# # Новый класс с несколькими эимбеддингами на челоека
# class FaceEmbedding(BaseFace):
#     __tablename__ = "face_embeddings"
#
#     id = Column(Integer, primary_key=True)
#     tg_id = mapped_column(BigInteger)  # ID пользователя, который добавил эмбеддинг
#     name = Column(String, nullable=False)  # Имя человека
#     embedding = Column(LargeBinary, nullable=False)  # Эмбеддинг
#     created_at = Column(DateTime, default=datetime.now)  # Время добавления эмбеддинга

#старый класс с одним эмеддингом
# class FaceEmbedding(BaseFace):
#     __tablename__ = "face_embeddings"
#
#     id = Column(Integer, primary_key=True)
#     tg_id = mapped_column(BigInteger)
#     name = Column(String, nullable=False)
#     embedding = Column(LargeBinary, nullable=False)


class Photo(BasePhoto):
    __tablename__ = "photos"

    id = Column(Integer, primary_key=True)
    file_path = Column(String(255), unique=True)  # Добавлен unique=True
    embedding_idx = Column(Integer)
    processed = Column(Boolean, default=False)
    processed_at = Column(DateTime, default=datetime.now)


async  def async_main():
    # Создание таблиц для FaceEmbedding в основной БД (engine)
    async with engine.begin() as conn:
        await conn.run_sync(BaseFace.metadata.create_all)

    # Создание таблиц для Photo в отдельной БД (enginephoto)
    async with enginephoto.begin() as conn:
        await conn.run_sync(BasePhoto.metadata.create_all)