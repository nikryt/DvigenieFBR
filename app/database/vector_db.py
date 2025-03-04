import faiss
import numpy as np
import os
from pathlib import Path


class VectorDB:
    def __init__(self, db_path: str = "faiss_index"):
        self.db_path = Path(db_path)
        self.index = None
        self.metadata = []
        self.dimension = 512  # Размерность эмбеддингов InceptionResnetV1

        # Создание/загрузка индекса
        if (self.db_path / "index.faiss").exists():
            self.index = faiss.read_index(str(self.db_path / "index.faiss"))
            with open(self.db_path / "metadata.txt", "r") as f:
                self.metadata = [line.strip() for line in f]
        else:
            self.db_path.mkdir(exist_ok=True)
            self.index = faiss.IndexFlatL2(self.dimension)

    async def insert_embedding(self, file_path: str, embedding: np.ndarray):
        # Преобразование в 2D массив
        embedding = np.expand_dims(embedding, axis=0).astype('float32')

        # Добавление в индекс
        self.index.add(embedding)
        self.metadata.append(file_path)

        # Сохранение на диск
        self._save()

    def search(self, query_embedding: np.ndarray, top_k: int = 1):
        query_embedding = np.expand_dims(query_embedding, axis=0).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)
        return [
            {
                "file_path": self.metadata[i],
                "distance": float(d)
            } for i, d in zip(indices[0], distances[0])
        ]

    def _save(self):
        faiss.write_index(self.index, str(self.db_path / "index.faiss"))
        with open(self.db_path / "metadata.txt", "w") as f:
            f.write("\n".join(self.metadata))