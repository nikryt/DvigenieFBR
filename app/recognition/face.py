import torch
import asyncio
import numpy as np
import cv2
import app.database.requests as rq

from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from app.database.vector_db import VectorDB
from dotenv import load_dotenv
from config import known_embeddings, known_embeddings_index




# Загрузка переменных из .env
load_dotenv()

# Инициализация моделей
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(
    keep_all=True,
    device=device,
    thresholds=[0.7, 0.8, 0.9], # Более строгие пороги
    min_face_size=100,  # Минимальный размер лица
    margin=20                     # Отступ вокруг лица
)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
vector_db = VectorDB()




# Функция распознавания фото которое прислали
async def recognize_face(image_path, threshold=0.56):
    # Загрузка изображения
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    # Обнаружение лиц
    faces = await asyncio.to_thread(mtcnn, image)
    if faces is None:
        return None

    # Извлечение эмбеддингов
    faces = faces.to(device)
    embeddings = resnet(faces).detach().cpu().numpy()

    # Сравнение с базой
    for embed in embeddings:
        for name, known_embed in known_embeddings.items():
            similarity = cosine_similarity([embed], [known_embed])[0][0]
            if similarity > threshold:
                return name

    return None

async def save_embedding(image_path: str, name: str, tg_id: str):
    """Сохранение эмбеддинга в БД"""
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    faces = mtcnn(image)
    if faces is None:
        return None

    face = faces[0].unsqueeze(0).to(device)
    embedding = resnet(face).detach().cpu().numpy()[0]
    known_embeddings[name] = embedding
    await rq.save_embedding(name, tg_id, embedding)  # Вызов функции из requests.py
    return embedding


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
