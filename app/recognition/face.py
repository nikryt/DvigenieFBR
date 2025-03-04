import torch
from sqlalchemy import select

from app.database.models import FaceEmbedding
from app.database.requests import async_session
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import cv2

# Инициализация моделей
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Глобальный кеш эмбеддингов (инициализируется при старте)
known_embeddings = {}

async def load_embeddings():
    """Загрузка эмбеддингов из БД"""
    global known_embeddings
    async with async_session() as db:
        result = await db.execute(select(FaceEmbedding))
        for row in result.scalars():
            known_embeddings[row.name] = np.frombuffer(row.embedding, dtype=np.float32)

# def load_embeddings():
#     embeddings = {}
#     for file in os.listdir("app/recognition/known_faces"):
#         if file.endswith(".npy"):
#             name = file.split(".")[0]
#             embeddings[name] = np.load(f"app/recognition/known_faces/{file}")
#     return embeddings


# # Загрузка известных эмбеддингов
# known_embeddings = load_embeddings()  # Загружаем при старте

# Функция распознавания
def recognize_face(image_path, threshold=0.56):
    # Загрузка изображения
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    # Обнаружение лиц
    faces = mtcnn(image)
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


async def save_embedding(image_path: str, name: str):
    """Сохранение эмбеддинга в БД"""
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    faces = mtcnn(image)
    if faces is None:
        return None

    face = faces[0].unsqueeze(0).to(device)
    embedding = resnet(face).detach().cpu().numpy()[0]

    # Обновляем БД и кеш
    async with async_session() as db:
        new_face = FaceEmbedding(name=name, embedding=embedding.tobytes())
        db.add(new_face)
        await db.commit()

    known_embeddings[name] = embedding
    return embedding

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