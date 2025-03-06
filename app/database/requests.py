import logging
import cv2
import sqlalchemy as sa
import numpy as np
import faiss


from config import PHOTOS_DIR, FAISS_INDEX_PATH, known_embeddings_index, FACE_EMBED_INDEX_PATH, \
    FACE_EMBEDDING_DIM, face_embeds_index, known_embeddings_names
from pathlib import Path
from sqlalchemy import select
from app.database.models import FaceEmbedding, async_session, Photo, async_session_photo, User

# отдельные таблицы для пользователей и их эмбеддингов
async def add_embedding(user_tg_id: int, user_name: str, embedding: np.ndarray):
    async with async_session() as session:
        # Найти или создать пользователя
        user = await session.execute(
            select(User).where(User.tg_id == user_tg_id)
        ).scalar_one_or_none()
        if not user:
            user = User(tg_id=user_tg_id, name=user_name)
            session.add(user)
            await session.commit()

        # Сохранить эмбеддинг
        embedding_bytes = embedding.tobytes()
        new_embedding = FaceEmbedding(user_id=user.id, embedding=embedding_bytes)
        session.add(new_embedding)
        await session.commit()


# # Новая функция добавления нескольких эмбеддингов
# async def add_embedding(name: str, tg_id: str, embedding: np.ndarray):
#     async with async_session() as session:
#         try:
#             embedding_bytes = embedding.tobytes()
#             new_face = FaceEmbedding(name=name, tg_id=tg_id, embedding=embedding_bytes)
#             session.add(new_face)
#             await session.commit()
#         except Exception as e:
#             await session.rollback()
#             raise e

# Старая функция добавления эмбеддинга
# async def add_embedding(name: str, tg_id: str, embedding: np.ndarray):
#     async with async_session() as db:
#         try:
#             # Используем select для поиска по имени
#             existing = (await db.execute(
#                 select(FaceEmbedding).where(FaceEmbedding.name == name)
#             )).scalar_one_or_none()
#
#             if existing:
#                 raise ValueError("Имя уже существует!")
#
#             embedding_bytes = embedding.tobytes()
#             new_face = FaceEmbedding(name=name, tg_id=tg_id, embedding=embedding_bytes)
#             db.add(new_face)
#             await db.commit()
#         except Exception as e:
#             await db.rollback()  # Асинхронный откат
#             raise e

async def get_embedding_by_name(name: str):
    async with async_session() as session:
        result = await session.execute(select(FaceEmbedding).filter_by(name=name))
        return result.scalar_one_or_none()


async def load_embeddings():
    """Загрузка эмбеддингов из БД и обновление FAISS индекса."""
    global known_embeddings
    known_embeddings = {}

    async with async_session() as session:
        result = await session.execute(
            select(User.name, FaceEmbedding.embedding)
            .join(FaceEmbedding.user)
        )

        for name, emb in result:
            embedding = np.frombuffer(emb, dtype=np.float32).copy()
            embedding = embedding / np.linalg.norm(embedding)

            if name not in known_embeddings:
                known_embeddings[name] = []
            known_embeddings[name].append(embedding)

            # Добавляем в FAISS индекс
            face_embeds_index.add(embedding.reshape(1, -1))
            known_embeddings_names.append(name)

    # Сохраняем обновленные данные
    faiss.write_index(face_embeds_index, FACE_EMBED_INDEX_PATH)
    np.save("face_names.npy", known_embeddings_names)

    return known_embeddings

# Работало, но начал обновлять с индексами faiss
# async def load_embeddings():
#     """Загрузка эмбеддингов из БД"""
#     global known_embeddings
#     known_embeddings = {}
#     async with async_session() as session:
#         result = await session.execute(select(FaceEmbedding))
#         rows = result.scalars().all()  # Получаем все записи
#
#         if not rows:
#             logging.error("Таблица face_embeddings пуста!")
#             return known_embeddings
#
#         for row in rows:
#             if row.name not in known_embeddings:
#                 known_embeddings[row.name] = []
#
#             embedding = np.frombuffer(row.embedding, dtype=np.float32)
#             embedding = embedding / np.linalg.norm(embedding)
#             known_embeddings[row.name].append(embedding)
#
#         logging.info(f"Загружено {len(known_embeddings)} уникальных имен с эмбеддингами.")
#         return known_embeddings

# async def load_embeddings():
#     """Загрузка эмбеддингов из БД"""
#     global known_embeddings
#     known_embeddings = {}
#     async with async_session() as session:
#         result = await session.execute(select(FaceEmbedding))
#         if not result.scalars().first():
#             logging.error("Таблица face_embeddings пуста!")
#             for row in result.scalars():
#                 # Сначала проверяем существование ключа
#                 if row.name not in known_embeddings:
#                     known_embeddings[row.name] = []
#
#                 # Обрабатываем эмбеддинг только один раз
#                 embedding = np.frombuffer(row.embedding, dtype=np.float32)
#                 embedding = embedding / np.linalg.norm(embedding)  # Нормализация
#                 known_embeddings[row.name].append(embedding)  # Добавляем нормализованный эмбеддинг
#
#         logging.info(f"Загружено {len(known_embeddings)} уникальных имен с эмбеддингами.")
#         return known_embeddings  # Добавьте возврат значения!


# # Старая функция загрузки 1 эмбеддинга
# async def load_embeddings():
#     """Загрузка эмбеддингов из БД"""
#     global known_embeddings
#     async with async_session() as db:
#         result = await db.execute(select(FaceEmbedding))
#         for row in result.scalars():
#             known_embeddings[row.name] = np.frombuffer(row.embedding, dtype=np.float32)


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
    """Сохранение эмбеддинга в БД с привязкой к пользователю."""
    async with async_session() as session:
        try:
            # Находим или создаем пользователя
            user = await session.execute(
                select(User).where(User.name == name)
            )
            user = user.scalar_one_or_none()

            if not user:
                # Создаем нового пользователя
                user = User(name=name, tg_id=tg_id)
                session.add(user)
                await session.commit()
                await session.refresh(user)

            # Сохраняем эмбеддинг
            embedding_bytes = embedding.tobytes()
            new_embedding = FaceEmbedding(
                user_id=user.id,
                embedding=embedding_bytes
            )
            session.add(new_embedding)
            await session.commit()

            # Обновляем кеш и индекс FAISS
            await update_cache_and_index(name, embedding)
            return embedding

        except Exception as e:
            await session.rollback()
            raise e

# # Новая функция добаления эмбеддингов. Теперь функция просто добавляет новый эмбеддинг
# # в базу данных, даже если имя уже существует.
# async def save_embedding(name: str, tg_id: str, embedding: np.ndarray):
#     """Сохранение эмбеддинга в БД"""
#     async with async_session() as session:
#         try:
#             embedding_bytes = embedding.tobytes()
#             new_face = FaceEmbedding(name=name, tg_id=tg_id, embedding=embedding_bytes)
#             session.add(new_face)
#             await session.commit()
#
#             # Обновляем кеш и индекс FAISS
#             await update_cache_and_index(name, embedding)
#         except Exception as e:
#             await session.rollback()
#             raise e

# Функция с добавления эмбеддингов с проверкой на существование имени в базе данныхэ
# async def save_embedding(name: str, tg_id: str, embedding: np.ndarray):
#     """Сохранение эмбеддинга в БД"""
#     async with async_session() as session:
#         try:
#             existing = (await session.execute(
#                 select(FaceEmbedding).where(FaceEmbedding.name == name)
#             )).scalar_one_or_none()
#
#             if existing:
#                 raise ValueError("Имя уже существует!")
#
#             embedding_bytes = embedding.tobytes()
#             new_face = FaceEmbedding(name=name, tg_id=tg_id, embedding=embedding_bytes)
#             session.add(new_face)
#             await session.commit()
#         except Exception as e:
#             await session.rollback()
#             raise e

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


async def get_all_names():
    """Получение уникальных имён пользователей из таблицы users."""
    async with async_session() as session:
        result = await session.execute(select(User.name).distinct())
        return [row[0] for row in result]

# Работало
# async def get_all_names():
#     """Возвращает список всех уникальных имён из базы данных."""
#     async with async_session() as session:
#         result = await session.execute(select(FaceEmbedding.name).distinct())
#         names = [row[0] for row in result]
#         return names


async def check_name_exists(name: str) -> bool:
    """Проверяет, существует ли имя в таблице users."""
    async with async_session() as session:
        result = await session.execute(select(User).where(User.name == name))
        return result.scalars().first() is not None

# Работало, начал добавлять FAISS и другую базу
# async def check_name_exists(name: str) -> bool:
#     """Проверяет, существует ли имя в базе данных."""
#     async with async_session() as session:
#         result = await session.execute(select(FaceEmbedding).where(FaceEmbedding.name == name))
#         return result.scalars().first() is not None

async def update_cache_and_index(name: str, embedding: np.ndarray):
    """Обновляет кеш эмбеддингов и индекс FAISS."""
    # Обновляем кеш
    if name not in known_embeddings:
        known_embeddings[name] = []
    known_embeddings[name].append(embedding)

    # Обновляем индекс FAISS
    face_embeds_index.add(embedding.reshape(1, -1))
    known_embeddings_names.append(name)

    # Сохраняем обновленный индекс и имена
    faiss.write_index(face_embeds_index, FACE_EMBED_INDEX_PATH)
    np.save("face_names.npy", known_embeddings_names)
