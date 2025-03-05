import cv2
import sqlalchemy as sa
import numpy as np
import faiss

from config import PHOTOS_DIR, FAISS_INDEX_PATH, known_embeddings_index
from pathlib import Path
from sqlalchemy import select, and_
from config import known_embeddings

from .models import FaceEmbedding, async_session, Photo, async_session_photo


async def add_embedding(name: str, tg_id: str, embedding: np.ndarray):
    async with async_session() as db:
        try:
            # Используем select для поиска по имени
            existing = (await db.execute(
                select(FaceEmbedding).where(FaceEmbedding.name == name)
            )).scalar_one_or_none()

            if existing:
                raise ValueError("Имя уже существует!")

            embedding_bytes = embedding.tobytes()
            new_face = FaceEmbedding(name=name, tg_id=tg_id, embedding=embedding_bytes)
            db.add(new_face)
            await db.commit()
        except Exception as e:
            await db.rollback()  # Асинхронный откат
            raise e

async def get_embedding_by_name(name: str):
    async with async_session() as db:
        result = await db.execute(select(FaceEmbedding).filter_by(name=name))
        return result.scalar_one_or_none()

async def load_embeddings():
    """Загрузка эмбеддингов из БД"""
    global known_embeddings
    async with async_session() as db:
        result = await db.execute(select(FaceEmbedding))
        for row in result.scalars():
            known_embeddings[row.name] = np.frombuffer(row.embedding, dtype=np.float32)


# Сохранение данных о файлах в папке
async def save_file_metadata(file_path: Path, person_name: str = None):
    async with async_session_photo() as session:
        # Используем first() вместо scalar_one_or_none()
        existing = await session.execute(sa.select(Photo).where(Photo.file_path == str(file_path)))
        if not existing.scalars().first():
            new_photo = Photo(
                file_path=str(file_path),
                person_name=person_name,
                processed=False
            )
            session.add(new_photo)
            await session.commit()




async def process_directory():
    """Обработка всех фотографий в целевой папке"""
    processed_files = set()

    # Получаем список уже обработанных файлов из БД
    async with async_session_photo() as session:
        result = await session.execute(select(Photo.file_path))
        processed_files = {row[0] for row in result}

    # Рекурсивный поиск изображений
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        for file_path in PHOTOS_DIR.rglob(ext):
            if str(file_path) in processed_files:
                continue

            # Обработка нового файла
            await process_image(file_path)


async def process_image(file_path: Path):
    """Обработка одного изображения и сохранение метаданных"""
    from app.recognition.face import mtcnn, resnet, device

    try:
        image = cv2.cvtColor(cv2.imread(str(file_path)), cv2.COLOR_BGR2RGB)
        faces = mtcnn(image)

        if faces is not None:
            faces = faces.to(device)
            embeddings = resnet(faces).detach().cpu().numpy()

            # Сохранение в Faiss
            await update_faiss_index(embeddings, str(file_path))

            # Сохранение метаданных
            await save_file_metadata(str(file_path))

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")


async def update_faiss_index(embeddings: np.ndarray, file_path: str):
    """Обновление индекса Faiss и связей с файлами"""
    index = faiss.read_index(FAISS_INDEX_PATH)

    # Добавляем эмбеддинги
    if index.ntotal == 0:
        index.add(embeddings)
    else:
        index.add(embeddings)

    # Сохраняем обновленный индекс
    faiss.write_index(index, FAISS_INDEX_PATH)

    # Сохраняем соответствия индексов и файлов
    async with async_session_photo() as session:
        # Удаляем старые записи перед добавлением новых
        await session.execute(
            sa.delete(Photo).where(Photo.file_path == file_path))
        # Добавляем только одну запись
        new_photo = Photo(
            file_path=file_path,
            embedding_idx=index.ntotal - len(embeddings),
            processed=True
        )
        session.add(new_photo)
        await session.commit()



async def save_embedding(name: str, tg_id: str, embedding: np.ndarray):
    """Сохранение эмбеддинга в БД"""
    async with async_session() as session:
        try:
            existing = (await session.execute(
                select(FaceEmbedding).where(FaceEmbedding.name == name)
            )).scalar_one_or_none()

            if existing:
                raise ValueError("Имя уже существует!")

            embedding_bytes = embedding.tobytes()
            new_face = FaceEmbedding(name=name, tg_id=tg_id, embedding=embedding_bytes)
            session.add(new_face)
            await session.commit()
        except Exception as e:
            await session.rollback()
            raise e

async def find_user_in_photos(user_embedding: np.ndarray, k=5):
    """Поиск пользователя в базе фотографий с использованием Faiss"""
    index = known_embeddings_index
    D, I = index.search(user_embedding.reshape(1, -1), k)

    async with async_session_photo() as session:
        result = await session.execute(
            select(Photo.file_path).where(Photo.embedding_idx.in_(I[0].tolist()))
        )
        return [row[0] for row in result.scalars()]

async def get_photos_by_name(name: str, k=100):
    """Поиск фотографий по имени человека"""
    async with async_session() as session:
        result = await session.execute(
            select(FaceEmbedding).where(FaceEmbedding.name == name)
        )
        face_embedding = result.scalar_one_or_none()

    if not face_embedding:
        return []

    user_embedding = np.frombuffer(face_embedding.embedding, dtype=np.float32)
    index = known_embeddings_index
    D, I = index.search(user_embedding.reshape(1, -1), k)

    async with async_session_photo() as session:
        result = await session.execute(
            select(Photo.file_path).where(
                Photo.embedding_idx.in_(I[0].tolist())
            )
        )
        return [row[0] for row in result]

