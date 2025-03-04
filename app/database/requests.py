from sqlalchemy import select
from sqlalchemy.orm import Session
from .models import Base, FaceEmbedding
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
import numpy as np

# Настройка подключения к БД
engine = create_async_engine('sqlite+aiosqlite:///faces.db')
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def add_embedding(name: str, embedding: np.ndarray):
    async with async_session() as db:
        try:
            # Используем select для поиска по имени
            existing = (await db.execute(
                select(FaceEmbedding).where(FaceEmbedding.name == name)
            )).scalar_one_or_none()

            if existing:
                raise ValueError("Имя уже существует!")

            embedding_bytes = embedding.tobytes()
            new_face = FaceEmbedding(name=name, embedding=embedding_bytes)
            db.add(new_face)
            await db.commit()
        except Exception as e:
            await db.rollback()  # Асинхронный откат
            raise e

async def get_embedding_by_name(name: str):
    async with async_session() as db:
        result = await db.execute(select(FaceEmbedding).filter_by(name=name))
        return result.scalar_one_or_none()


