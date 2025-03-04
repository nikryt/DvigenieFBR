from pathlib import Path

import faiss
import numpy as np

# Глобальный кеш эмбеддингов (инициализируется при старте)
known_embeddings = {}

# Конфигурация обработки папки
PHOTOS_DIR = Path("user_photos")
FAISS_INDEX_PATH = "embeddings.faiss"

# Инициализация индекса Faiss
def init_faiss_index(dim=512):
    return faiss.IndexFlatL2(dim)

try:
    known_embeddings_index = faiss.read_index(FAISS_INDEX_PATH)
except:
    known_embeddings_index = init_faiss_index()
    faiss.write_index(known_embeddings_index, FAISS_INDEX_PATH)