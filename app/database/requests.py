import numpy as np
from sqlalchemy import select


from .models import FaceEmbedding, async_session


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


