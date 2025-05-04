import json
from pathlib import Path
import zipfile
from io import BytesIO
from datetime import datetime
from app.database.requests import (
    get_all_users_with_embeddings,
    create_user,
    get_user_by_tg_id,
    create_embedding,
    delete_user, get_user_by_name
)


async def export_user_data(user_name: str = None) -> BytesIO:
    """
    Экспорт данных пользователя(ей) в BytesIO объект с ZIP-архивом
    :param user_name: Если None - экспорт всех пользователей
    """
    buffer = BytesIO()

    if user_name:
        users = [await get_user_by_name(user_name)]
    else:
        users = await get_all_users_with_embeddings()

    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Добавляем метаданные
        meta = {
            "export_type": "single" if user_name else "full",
            "exported_at": datetime.now().isoformat(),
            "users_count": len(users),
            "embeddings_count": sum(len(u.embeddings) for u in users if u)
        }
        zipf.writestr("meta.json", json.dumps(meta, indent=2))

        for user in users:
            if not user:
                continue

            user_dir = f"users/{user.id}"
            user_data = {
                "id": user.id,
                "tg_id": user.tg_id,
                "name": user.name,
                "created_at": user.created_at.isoformat(),
                "embeddings": len(user.embeddings)
            }

            # Записываем данные пользователя
            zipf.writestr(f"{user_dir}/user.json", json.dumps(user_data, indent=2))

            # Сохраняем эмбеддинги
            for emb in user.embeddings:
                emb_path = f"{user_dir}/embeddings/{emb.id}.bin"
                zipf.writestr(emb_path, emb.embedding)

    buffer.seek(0)
    return buffer


async def export_data(export_path: Path = Path("face_data_export.zip")):
    """Экспорт данных в ZIP-архив"""
    buffer = BytesIO()

    users = await get_all_users_with_embeddings()

    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Метаданные
        meta = {
            "version": 1,
            "exported_at": datetime.now().isoformat(),
            "users_count": len(users),
            "embeddings_count": sum(len(u.embeddings) for u in users)
        }
        zipf.writestr("meta.json", json.dumps(meta, indent=2))

        # Данные пользователей
        for user in users:
            user_dir = f"users/{user.id}"
            user_data = {
                "id": user.id,
                "tg_id": user.tg_id,
                "name": user.name,
                "created_at": user.created_at.isoformat(),
                "embeddings_count": len(user.embeddings)
            }
            zipf.writestr(f"{user_dir}/user.json", json.dumps(user_data, indent=2))

            # Эмбеддинги
            for emb in user.embeddings:
                emb_data = {
                    "id": emb.id,
                    "user_id": user.id,
                    "created_at": emb.created_at.isoformat()
                }
                zipf.writestr(f"{user_dir}/embeddings/{emb.id}.bin", emb.embedding)
                zipf.writestr(f"{user_dir}/embeddings/{emb.id}.json", json.dumps(emb_data))

    with open(export_path, 'wb') as f:
        f.write(buffer.getvalue())


async def import_data(import_path: Path, conflict_resolve: str = "skip"):
    """Импорт данных из ZIP-архива"""
    user_map = {}

    with zipfile.ZipFile(import_path, 'r') as zipf:
        meta = json.loads(zipf.read("meta.json"))

        # Обработка пользователей
        for entry in zipf.namelist():
            if not entry.endswith("/user.json"):
                continue

            user_data = json.loads(zipf.read(entry))
            original_id = user_data['id']

            # Проверка существующего пользователя
            existing = await get_user_by_tg_id(user_data['tg_id'])

            if existing:
                if conflict_resolve == "skip":
                    continue
                elif conflict_resolve == "replace":
                    await delete_user(existing.id)

            # Создание пользователя
            new_user = await create_user(user_data)
            user_map[original_id] = new_user.id

            # Импорт эмбеддингов
            emb_dir = f"users/{original_id}/embeddings/"
            for emb_entry in zipf.namelist():
                if emb_entry.startswith(emb_dir) and emb_entry.endswith(".bin"):
                    emb_data = {
                        "user_id": new_user.id,
                        "embedding": zipf.read(emb_entry),
                        "created_at": user_data['created_at']
                    }
                    await create_embedding(emb_data)