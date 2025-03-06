import logging
import os
from datetime import datetime

import cv2
import sqlalchemy as sa
import numpy as np
import faiss
from sqlalchemy.orm import selectinload

from config import (PHOTOS_DIR, FAISS_INDEX_PATH, known_embeddings_index, FACE_EMBED_INDEX_PATH,
                    face_embeds_index, known_embeddings_names)

from sqlalchemy import select
from app.database.models import FaceEmbedding, async_session, Photo, async_session_photo, User
from app.recognition.face import mtcnn, resnet, device




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

    # Очистка индекса
    face_embeds_index.reset()
    known_embeddings_names.clear()

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

            # Логируем добавление
            logging.info(f"Добавлен эмбеддинг для {name} в индекс FAISS")

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


async def process_directory():
    """Обработка директории с фотографиями и добавление в FAISS"""
    try:
        # Получаем список всех файлов изображений с корректными путями
        image_paths = []
        for root, _, files in os.walk(PHOTOS_DIR):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Формируем относительный путь БЕЗ дублирования папки
                    rel_path = os.path.relpath(
                        os.path.join(root, file),
                        start=str(PHOTOS_DIR)
                    )
                    image_paths.append(rel_path)

        # Получаем уже обработанные файлы
        async with async_session_photo() as session:
            result = await session.execute(select(Photo.file_path))
            # Исправлено: используем скалярные значения напрямую
            processed = {row for row in result.scalars()}

        new_images = [p for p in image_paths if p not in processed]
        total_faces = 0

        for img_rel_path in new_images:
            full_path = os.path.join(PHOTOS_DIR, img_rel_path)
            try:
                # Проверка существования файла
                if not os.path.exists(full_path):
                    logging.error(f"Файл отсутствует: {full_path}")
                    continue

                # Чтение файла с обработкой ошибок
                image = cv2.imread(full_path)
                if image is None:
                    logging.error(f"Ошибка чтения файла: {full_path}")
                    continue

                # Обнаружение лиц
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                faces = mtcnn(image_rgb)

                if faces is None:
                    continue

                # Генерация эмбеддингов
                faces_tensor = faces.to(device)
                embeddings = resnet(faces_tensor).detach().cpu().numpy()
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

                if embeddings.shape[0] == 0:
                    continue

                # Добавление в FAISS и базу данных
                async with async_session_photo() as session:
                    # Добавляем эмбеддинги в индекс
                    start_idx = known_embeddings_index.ntotal
                    known_embeddings_index.add(embeddings.astype(np.float32))

                    # Сохраняем записи в базу
                    for face_idx in range(embeddings.shape[0]):
                        photo = Photo(
                            file_path=img_rel_path,
                            embedding_idx=start_idx + face_idx,
                            face_index=face_idx,
                            processed=True,
                            processed_at=datetime.now()
                        )
                        session.add(photo)
                    await session.commit()

                total_faces += embeddings.shape[0]

            except Exception as e:
                logging.error(f"Ошибка обработки {full_path}: {str(e)}", exc_info=True)

        # Сохранение индекса
        faiss.write_index(known_embeddings_index, FAISS_INDEX_PATH)
        return total_faces

    except Exception as e:
        logging.error(f"Фатальная ошибка: {str(e)}", exc_info=True)
        return 0


# Рекомендуемые параметры:
#
# Для точного поиска: threshold=0.65, k_nearest=30
# Для широкого поиска: threshold=0.55, k_nearest=100

async def get_photos_by_name(name: str, threshold: float = 0.60, k_nearest: int = 30) -> list[str]:
    """Поиск фотографий по имени с исправлением ошибок индексов"""
    try:
        # 1. Получение пользователя и его эмбеддингов
        async with async_session() as session:
            user = await session.scalar(
                select(User)
                .options(selectinload(User.embeddings))
                .where(User.name == name)
            )
            if not user:
                logging.warning(f"Пользователь '{name}' не найден")
                return []

            # 2. Валидация эмбеддингов
            valid_embeddings = [
                np.frombuffer(emb.embedding, dtype=np.float32)
                for emb in user.embeddings
                if isinstance(emb.embedding, bytes)
                   and np.frombuffer(emb.embedding, dtype=np.float32).size == 512
            ]
            logging.info(f"Найдено валидных эмбеддингов: {len(valid_embeddings)}")

        # 3. Поиск в FAISS с явным приведением типов
        all_matches = set()
        for embed in valid_embeddings:
            embed_norm = embed / np.linalg.norm(embed)
            query = embed_norm.astype(np.float32).reshape(1, -1)
            distances, indices = known_embeddings_index.search(query, k_nearest)

            # Явное преобразование np.int64 в int
            matches = [int(idx) for idx, dist in zip(indices[0], distances[0])
                       if (1 - 0.5 * dist) >= threshold]
            all_matches.update(matches)
            logging.info(f"Найдено совпадений: {matches}")

        # 4. Запрос к базе данных с проверкой существования индексов
        async with async_session_photo() as session:
            if not all_matches:
                return []

            # Преобразование в список int
            matches_list = list(map(int, all_matches))

            # Проверка существования индексов в базе
            existing_indices = await session.scalars(
                select(Photo.embedding_idx)
                .where(Photo.embedding_idx.in_(matches_list))
            )
            existing_indices = set(existing_indices.all())

            # Получение путей только для существующих индексов
            result = await session.scalars(
                select(Photo.file_path)
                .where(Photo.embedding_idx.in_(existing_indices))
                .distinct()
            )
            file_names = result.unique().all()

            # Фильтрация существующих файлов
            valid_photos = []
            for name in file_names:
                if (PHOTOS_DIR / name).exists():
                    valid_photos.append(name)
                else:
                    logging.error(f"Файл отсутствует: {name}")

            logging.info(f"Найдено фотографий: {len(valid_photos)}")
            return valid_photos

    except Exception as e:
        logging.error(f"Критическая ошибка: {str(e)}", exc_info=True)
        return []

# Сохранение данных о файлах в папке
# async def save_file_metadata(file_path: Path, person_name: str = None):
#     async with async_session_photo() as session:
#         # Используем first() вместо scalar_one_or_none()
#         existing = await session.execute(sa.select(Photo).where(Photo.file_path == str(file_path)))
#         if not existing.scalars().first():
#             new_photo = Photo(
#                 file_path=str(file_path),
#                 person_name=person_name,
#                 processed=False
#             )
#             session.add(new_photo)
#             await session.commit()




# async def process_directory():
#     """Обработка всех фотографий в целевой папке"""
#     processed_files = set()
#
#     # Получаем список уже обработанных файлов из БД
#     async with async_session_photo() as session:
#         result = await session.execute(select(Photo.file_path))
#         processed_files = {row[0] for row in result}
#
#     # Рекурсивный поиск изображений
#     for ext in ['*.jpg', '*.jpeg', '*.png']:
#         for file_path in PHOTOS_DIR.rglob(ext):
#             # Сохраняем относительный путь
#             relative_path = file_path.relative_to(PHOTOS_DIR)
#             if str(relative_path) in processed_files:
#                 continue
#
#             await process_image(relative_path)  # Передаем относительный путь


# async def process_image(relative_path: Path):
#     """Обработка одного изображения и сохранение метаданных"""
#     try:
#         # Формируем абсолютный путь для обработки
#         abs_path = PHOTOS_DIR / relative_path
#         image = cv2.cvtColor(cv2.imread(str(abs_path)), cv2.COLOR_BGR2RGB)
#         faces = mtcnn(image)
#
#         if faces is not None:
#             faces = faces.to(device)
#             embeddings = resnet(faces).detach().cpu().numpy()
#
#             # Сохранение в FAISS и метаданных для каждого лица
#             for face_index, embedding in enumerate(embeddings):
#                 await update_faiss_index(embedding.reshape(1, -1), str(relative_path), face_index)
#
#     except Exception as e:
#         print(f"Error processing {relative_path}: {str(e)}")

# async def process_image(file_path: Path):
#     """Обработка одного изображения и сохранение метаданных"""
#     from app.recognition.face import mtcnn, resnet, device
#
#     try:
#         image = cv2.cvtColor(cv2.imread(str(file_path)), cv2.COLOR_BGR2RGB)
#         faces = mtcnn(image)
#
#         if faces is not None:
#             faces = faces.to(device)
#             embeddings = resnet(faces).detach().cpu().numpy()
#
#             # Сохранение в Faiss
#             await update_faiss_index(embeddings, str(file_path))
#
#             # Сохранение метаданных
#             await save_file_metadata(str(file_path))
#
#     except Exception as e:
#         print(f"Error processing {file_path}: {str(e)}")

# async def update_faiss_index(embedding: np.ndarray, relative_path: str, face_index: int):
#     """Обновление индекса Faiss и связей с файлами для каждого лица"""
#     try:
#         async with async_session_photo() as session:
#             new_photo = Photo(
#                 file_path=str(relative_path),  # Сохраняем относительный путь как строку
#                 embedding_idx=index.ntotal - 1,
#                 face_index=face_index,
#                 processed=True
#             )
#             session.add(new_photo)
#             await session.commit()
#             logging.info(f"Добавлен эмбеддинг для файла {relative_path}")
#     except Exception as e:
#         print(f"Error processing {relative_path}: {str(e)}")


#
# async def update_faiss_index(embeddings: np.ndarray, file_path: str):
#     """Обновление индекса Faiss и связей с файлами"""
#     index = faiss.read_index(FAISS_INDEX_PATH)
#
#     # Добавляем эмбеддинги
#     if index.ntotal == 0:
#         index.add(embeddings)
#     else:
#         index.add(embeddings)
#
#     # Сохраняем обновленный индекс
#     faiss.write_index(index, FAISS_INDEX_PATH)
#
#     # Сохраняем соответствия индексов и файлов
#     async with async_session_photo() as session:
#         # Удаляем старые записи перед добавлением новых
#         await session.execute(
#             sa.delete(Photo).where(Photo.file_path == file_path))
#         # Добавляем только одну запись
#         new_photo = Photo(
#             file_path=file_path,
#             embedding_idx=index.ntotal - len(embeddings),
#             processed=True
#         )
#         session.add(new_photo)
#         await session.commit()
#
#
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
            select(Photo.file_path).where(Photo.embedding_idx.in_(I[0].tolist())))

        paths = [row[0] for row in result.scalars()]

        # Преобразуем относительные пути в абсолютные
        return [str(PHOTOS_DIR / path) for path in paths]


# async def get_photos_by_name(name: str):
#     """Поиск фотографий по имени пользователя."""
#     async with async_session() as session:
#         user_result = await session.execute(
#             select(User).where(User.name == name))
#         user = user_result.scalar_one_or_none()
#
#         if not user:
#             return []
#
#         embeddings_result = await session.execute(
#             select(FaceEmbedding).where(FaceEmbedding.user_id == user.id))
#         embeddings = embeddings_result.scalars().all()
#
#         photo_paths = []
#         for emb in embeddings:
#             embedding = np.frombuffer(emb.embedding, dtype=np.float32)
#             paths = await find_user_in_photos(embedding)
#             photo_paths.extend(paths)
#             # Исправлено логирование
#             logging.info(f"Found paths: {paths}")
#
#         return list(set(photo_paths))


# async def get_photos_by_name(name: str, k=100, similarity_threshold=0.1):
#     """Поиск фотографий по имени пользователя."""
#     try:
#         # Поиск пользователя в БД users
#         async with async_session() as session_users:
#             user = await session_users.execute(
#                 select(User).where(func.lower(User.name) == name.lower())
#             )
#             user = user.scalar_one_or_none()
#             if not user:
#                 return []
#
#             # Получение эмбеддингов пользователя
#             embeddings = await session_users.execute(
#                 select(FaceEmbedding).where(FaceEmbedding.user_id == user.id)
#             )
#             embeddings = embeddings.scalars().all()
#             if not embeddings:
#                 return []
#
#         # Загрузка индекса фотографий
#         index = faiss.read_index(FAISS_INDEX_PATH)
#         photo_paths = []
#
#         # Поиск в индексе FAISS для каждого эмбеддинга пользователя
#         for emb in embeddings:
#             user_embedding = np.frombuffer(emb.embedding, dtype=np.float32)
#             D, I = index.search(user_embedding.reshape(1, -1), k)
#
#             # Получение путей из БД photos
#             async with async_session_photo() as session_photos:
#                 for idx in I[0]:
#                     result = await session_photos.execute(
#                         select(Photo.file_path).where(Photo.embedding_idx == idx)
#                     )
#                     photo = result.scalar_one_or_none()
#                     if photo:
#                         photo_paths.append(photo.replace("\\", "/"))
#
#         return list(set(photo_paths))
#     except Exception as e:
#         logging.error(f"Ошибка: {str(e)}")
#         return []
#
# async def get_photos_by_name(name: str, k=100, similarity_threshold=0.3):
#     """Поиск фотографий по имени человека с использованием новой модели таблиц и фильтрацией по сходству."""
#     logging.info(f"Поиск фото для имени: {name}")
#     async with async_session() as session:
#         # Находим пользователя по имени
#         result = await session.execute(
#             select(User).where(User.name == name)
#         )
#         user = result.scalar_one_or_none()
#
#         if not user:
#             logging.info(f"Пользователь с именем {name} не найден.")
#             return []  # Если пользователь не найден, возвращаем пустой список
#
#         # Получаем все эмбеддинги для этого пользователя
#         result = await session.execute(
#             select(FaceEmbedding).where(FaceEmbedding.user_id == user.id)
#         )
#         embeddings = result.scalars().all()
#
#         if not embeddings:
#             logging.info(f"Эмбеддинги для пользователя {name} не найдены.")
#             return []  # Если эмбеддинги не найдены, возвращаем пустой список
#
#         # Преобразуем эмбеддинги в массив numpy
#         user_embeddings = [np.frombuffer(embedding.embedding, dtype=np.float32) for embedding in embeddings]
#
#         # Ищем фотографии в базе фотографий с использованием Faiss
#         photo_paths = []
#         for embedding in user_embeddings:
#             index = faiss.read_index(FAISS_INDEX_PATH)  # Используем индекс фотографий
#             D, I = index.search(embedding.reshape(1, -1), k)
#
#             # Фильтруем результаты по порогу сходства
#             for i, distance in zip(I[0], D[0]):
#                 # similarity = 1 - distance  # Преобразуем расстояние L2 в сходство
#                 similarity = 1 / (1 + distance)  # Преобразование L2 в сходство (0..1)
#                 if similarity > similarity_threshold:
#                     async with async_session_photo() as session_photo:
#                         result = await session_photo.execute(
#                             select(Photo.file_path).where(
#                                 Photo.embedding_idx == i
#                             )
#                         )
#                         photo = result.scalar_one_or_none()
#                         if photo:
#                             photo_paths.append(photo)
#                             logging.info(f"Найдена фотография: {photo}, сходство: {similarity}")
#
#         return list(set(photo_paths))  # Убираем дубликаты

async def get_all_names():
    """Получение уникальных имён пользователей"""
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
