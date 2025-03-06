import logging
from pathlib import Path

import faiss
import numpy as np

# Глобальный кеш эмбеддингов (инициализируется при старте)
known_embeddings = {}

def update_known_embeddings(new_embeddings):
    global known_embeddings
    # Теперь grouped by user name
    known_embeddings = {
        name: [emb for embs in new_embeddings.values() for emb in embs]
        for name in new_embeddings
    }
    logging.info(f"Обновлено эмбеддингов: {len(known_embeddings)}")

# def update_known_embeddings(new_embeddings):
#     global known_embeddings
#     known_embeddings = new_embeddings
#     logging.info(f"Обновлено эмбеддингов: {len(known_embeddings)}")

def get_known_embeddings():
    return known_embeddings

# Конфигурация обработки папки
PHOTOS_DIR = Path("user_photos")
FAISS_INDEX_PATH = "embeddings.faiss"

# Добавьте новую конфигурацию для индекса эмбеддингов лиц
FACE_EMBED_INDEX_PATH = "face_embeds.faiss"
FACE_EMBEDDING_DIM = 512  # Размерность эмбеддинга

# Инициализация индекса для эмбеддингов лиц
try:
    face_embeds_index = faiss.read_index(FACE_EMBED_INDEX_PATH)
    known_embeddings_names = np.load("face_names.npy", allow_pickle=True).tolist()
except:
    face_embeds_index = faiss.IndexFlatL2(FACE_EMBEDDING_DIM)
    known_embeddings_names = []
    faiss.write_index(face_embeds_index, FACE_EMBED_INDEX_PATH)
    np.save("face_names.npy", known_embeddings_names)

# Инициализация индекса Faiss
def init_faiss_index(dim=512):
    return faiss.IndexFlatL2(dim)

try:
    known_embeddings_index = faiss.read_index(FAISS_INDEX_PATH)
except:
    known_embeddings_index = init_faiss_index()
    faiss.write_index(known_embeddings_index, FAISS_INDEX_PATH)