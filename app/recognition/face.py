import torch
import asyncio
import logging
import cv2
import numpy as np
import app.database.requests as rq


from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from config import face_embeds_index, known_embeddings_names, get_known_embeddings, known_embeddings



# Загрузка переменных из .env
load_dotenv()

# Инициализация моделей
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(
    keep_all=True,
    device=device,
    thresholds=[0.6, 0.7, 0.7],  # Более мягкие пороги
    min_face_size=40,  # Уменьшенный размер
    margin=10   # Отступ вокруг лица
    # thresholds=[0.7, 0.8, 0.9], # Более строгие пороги
    # min_face_size=100,  # Минимальный размер лица
    # margin=20     # Отступ вокруг лица
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)






# Вот улучшенная функция распознавания для файла face.py,
# которая корректно обрабатывает несколько эмбеддингов для одного человека и использует FAISS для быстрого поиска:
async def recognize_face(image_path, threshold=0.56):
    """Распознавание лица на фотографии с использованием нескольких эмбеддингов."""
    # Получаем актуальные эмбеддинги
    known_embeddings = get_known_embeddings()

    # Проверка наличия эмбеддингов
    if not known_embeddings:
        logging.error("База эмбеддингов пуста!")
        return None

    # Логируем первые 3 имени из базы
    sample_names = list(known_embeddings.keys())[:3]
    logging.info(f"Пример имен в базе: {sample_names}")

    logging.info(f"Начало распознавания лица для файла: {image_path}")
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    faces = await asyncio.to_thread(mtcnn, image)
    if faces is None:
        return None

    logging.info(f"Обнаружено {len(faces)} лиц на фотографии.")
    faces = faces.to(device)
    embeddings = resnet(faces).detach().cpu().numpy()

    # Нормализация эмбеддингов
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Поиск ближайших соседей в FAISS
    distances, indices = face_embeds_index.search(embeddings, 5)  # Ищем 5 ближайших

    matches = []
    for i in range(len(embeddings)):
        best_sim = 0
        best_name = None

        # Ищем максимальное сходство среди всех совпадений для лица
        for d, idx in zip(distances[i], indices[i]):
            if idx >= len(known_embeddings_names):
                continue

            similarity = 1 - 0.5 * d  # Преобразуем расстояние в сходство
            if similarity > best_sim and similarity > threshold:
                best_sim = similarity
                best_name = known_embeddings_names[idx]

        if best_name:
            matches.append((best_name, best_sim))

    # Группируем результаты по имени
    results = {}
    for name, sim in matches:
        if name not in results or sim > results[name]:
            results[name] = sim

    # Сортируем результаты по убыванию сходства
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    # Возвращаем результаты, если они есть
    return sorted_results if sorted_results else None

#
# # Добавляю FAISS  в ноывой функции
# # Новая функция распознования с несколькими эмбеддингами
# # Функция распознавания фото которое прислали
# async def recognize_face(image_path, threshold=0.56):
#     """Распознавание лица на фотографии."""
#     # Получаем актуальные эмбеддинги
#     known_embeddings = get_known_embeddings()
#
#     # Проверка наличия эмбеддингов
#     if not known_embeddings:
#         logging.error("База эмбеддингов пуста!")
#         return None
#
#     # Логируем первые 3 имени из базы
#     sample_names = list(known_embeddings.keys())[:3]
#     logging.info(f"Пример имен в базе: {sample_names}")
#
#     logging.info(f"Начало распознавания лица для файла: {image_path}")
#     image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
#     faces = await asyncio.to_thread(mtcnn, image)
#     if faces is None:
#         return None
#
#     logging.info(f"Обнаружено {len(faces)} лиц на фотографии.")
#     faces = faces.to(device)
#     embeddings = resnet(faces).detach().cpu().numpy()
#     # Нормализация
#     embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
#
#     best_match = None
#     best_similarity = threshold
#
#     for embed in embeddings:
#         # Преобразуем embed в двумерный массив
#         embed_2d = embed.reshape(1, -1)  # Преобразуем в форму (1, n)
#
#         for name, known_embeds in known_embeddings.items():
#             for known_embed in known_embeds:
#                 known_embed = known_embed / np.linalg.norm(known_embed)  # Нормализация
#                 # Преобразуем known_embed в двумерный массив
#                 known_embed_2d = known_embed.reshape(1, -1)  # Преобразуем в форму (1, n)
#
#                 # Вычисляем косинусное сходство
#                 similarity = cosine_similarity(embed_2d, known_embed_2d)[0][0]
#                 logging.info(f"Сравнение с {name}: similarity={similarity}, max_similarity={best_similarity}")# Логируем сходство
#
#                 if similarity > best_similarity:
#                     best_similarity = similarity
#                     best_match = name
#     logging.info(f"Лучшее совпадение: {best_match} с косинусным сходством {best_similarity}")
#     return best_match

# Старая функция распознования с 1 эмбеддингом
# # Функция распознавания фото которое прислали
# async def recognize_face(image_path, threshold=0.56):
#     # Загрузка изображения
#     image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
#
#     # Обнаружение лиц
#     faces = await asyncio.to_thread(mtcnn, image)
#     if faces is None:
#         return None
#
#     # Извлечение эмбеддингов
#     faces = faces.to(device)
#     embeddings = resnet(faces).detach().cpu().numpy()
#
#     # Сравнение с базой
#     for embed in embeddings:
#         for name, known_embed in known_embeddings.items():
#             similarity = cosine_similarity([embed], [known_embed])[0][0]
#             if similarity > threshold:
#                 return name
#
#     return None

async def save_embedding(image_path: str, name: str, tg_id: str):
    """Извлечение и сохранение эмбеддинга."""
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    faces = mtcnn(image)
    if faces is None:
        return None

    face = faces[0].unsqueeze(0).to(device)
    embedding = resnet(face).detach().cpu().numpy()[0]
    embedding = embedding / np.linalg.norm(embedding)  # Нормализация
    logging.info(f"Эмбеддинг для {name} успешно извлечён.")

    # Проверка на дубликаты
    if name in known_embeddings:
        similarities = []
        for existing in known_embeddings[name]:
            similarities.append(cosine_similarity([embedding], [existing])[0][0])

        if max(similarities) > 0.92:  # Порог дубликата
            logging.warning(f"Duplicate embedding for {name} detected")
            return None

    # Сохраняем эмбеддинг через requests.py
    return await rq.save_embedding(name, tg_id, embedding)

# Работало, начал новую базу
# async def save_embedding(image_path: str, name: str, tg_id: str):
#     """Сохранение эмбеддинга в БД"""
#     image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
#     faces = mtcnn(image)
#     if faces is None:
#         logging.warning("Лицо не обнаружено на фотографии.")
#         return None
#
#     face = faces[0].unsqueeze(0).to(device)
#     embedding = resnet(face).detach().cpu().numpy()[0]  # Извлекаем эмбеддинг
#     embedding = embedding / np.linalg.norm(embedding)  # Нормализация
#     logging.info(f"Эмбеддинг для {name} успешно извлечён.")
#
#     # Проверка на дубликаты
#     if name in known_embeddings:
#         similarities = []
#         for existing in known_embeddings[name]:
#             similarities.append(cosine_similarity([embedding], [existing])[0][0])
#
#         if max(similarities) > 0.92:  # Порог дубликата
#             logging.warning(f"Duplicate embedding for {name} detected")
#             return None
#
#     # Сохраняем нормализованный эмбеддинг
#     await rq.save_embedding(name, tg_id, embedding)
#     return embedding


async def find_user_in_photos(user_embedding: np.ndarray, k=5):
    """Поиск пользователя в базе фотографий с использованием Faiss"""
    return await rq.find_user_in_photos(user_embedding, k)  # Вызов функции из requests.py

async def get_photos_by_name(name: str, k=100):
    """Поиск фотографий по имени человека"""
    return await rq.get_photos_by_name(name, k)  # Вызов функции из requests.py

# def load_embeddings():
#     embeddings = {}
#     for file in os.listdir("app/recognition/known_faces"):
#         if file.endswith(".npy"):
#             name = file.split(".")[0]
#             embeddings[name] = np.load(f"app/recognition/known_faces/{file}")
#     return embeddings


# # Загрузка известных эмбеддингов
# known_embeddings = load_embeddings()  # Загружаем при старте


# def save_embedding(image_path, name):  # Добавляем параметр name
#     image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
#     faces = mtcnn(image)
#     if faces is None:
#         return None
#
#     face = faces[0].unsqueeze(0).to(device)
#     embedding = resnet(face).detach().cpu().numpy()[0]
#
#     # Сохранение в файл с правильным именем
#     np.save(f"app/recognition/known_faces/{name}.npy", embedding)  # Используем переданное имя
#
#     known_embeddings[name] = embedding  # Обновляем кеш
#     return embedding  # Для сохранения в БД
