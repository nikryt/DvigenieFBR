import asyncio
import os
import cv2
import logging
from pathlib import Path
from aiogram import Router, Bot, F
from aiogram.types import Message, CallbackQuery, FSInputFile, InputMediaPhoto
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from sqlalchemy import select

from app.database.models import User, async_session
# from app.recognition.face import recognize_face, save_embedding, mtcnn, get_photos_by_name
# from config import known_embeddings


import app.database.requests as rq
import app.keyboards as kb
import app.recognition.face as fc

router = Router()
logger = logging.getLogger(__name__)

# Создаем необходимые директории
Path("downloads").mkdir(exist_ok=True)
Path("app/recognition/known_faces").mkdir(parents=True, exist_ok=True)


class AddFaceState(StatesGroup):
    waiting_for_name_selection = State()
    waiting_for_new_photo = State()  # Новое состояние для добавления новых фото
    waiting_for_photo = State()
    waiting_for_name = State()
    confirmation = State()

# Добавим в класс состояний
class MainState(StatesGroup):
    recognizing = State()
    adding_face = AddFaceState()



async def download_photo(bot: Bot, file_id: str, prefix: str = "") -> str:
    """Скачивает фото и возвращает путь к файлу"""
    try:
        file = await bot.get_file(file_id)
        file_path = file.file_path
        download_path = Path("downloads") / f"{prefix}{file_id}.jpg"

        await bot.download_file(file_path, str(download_path))
        return str(download_path)

    except Exception as e:
        logger.error(f"Ошибка при скачивании файла: {str(e)}")
        raise


async def validate_photo(image_path: str) -> bool:
    """Проверяет фото на наличие одного лица"""
    try:
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        faces = fc.mtcnn(image)

        if faces is None or len(faces) == 0:
            logger.warning("Лицо не обнаружено")
            return False
        if len(faces) > 1:
            logger.warning("Обнаружено несколько лиц")
            return False
        return True

    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {str(e)}")
        return False


@router.message(Command("start"))
async def start_handler(message: Message, state: FSMContext):
    """Обработчик команды /start"""
    await message.answer(
        "Привет! Отправь мне фото человека, и я попробую его распознать.",
        reply_markup=kb.get_start_keyboard(),
    )
    await state.set_state(MainState.recognizing)


@router.message(lambda message: message.text == "Распознать лицо")
async def recognize_handler(message: Message, state: FSMContext):
    """Запуск процесса распознавания"""
    await message.answer("Отправьте фото для распознавания")
    await state.set_state(MainState.recognizing)


@router.message(MainState.recognizing, lambda message: message.photo)
async def recognition_photo_handler(message: Message, bot: Bot, state: FSMContext):
    """Обработка фото для распознавания"""
    try:
        photo = message.photo[-1]
        download_path = await download_photo(bot, photo.file_id, "recog_")

        if result := await fc.recognize_face(download_path):
            response = "Распознанные лица:\n" + "\n".join(
                [f"{name} ({sim:.2%})" for name, sim in result]
            )
            await message.answer(response)
        else:
            await message.answer("Не удалось распознать людей.")
        #заменил блок на новый для работы с FAISS
        # if result := await fc.recognize_face(download_path):
        #     await message.answer(f"На фотографии есть: {result}")
        # else:
        #     await message.answer("Не удалось распознать человека.")

        os.remove(download_path)
        await state.clear()

    except Exception as e:
        logger.error(f"Recognition error: {str(e)}")
        await message.answer("Ошибка распознавания. Попробуйте еще раз.")
        await state.clear()


@router.message(lambda message: message.text == "Добавить эмбеддинги")
async def add_embeddings_handler(message: Message, state: FSMContext):
    # Получаем список всех имён из базы данных
    names = await rq.get_all_names()

    if not names:
        await message.answer("В базе данных пока нет лиц.")
        return

    # Создаём клавиатуру с именами
    keyboard = kb.get_names_keyboard(names)

    await message.answer("Выберите имя человека, для которого хотите добавить эмбеддинги:", reply_markup=keyboard)
    await state.set_state(AddFaceState.waiting_for_name_selection)

@router.message(AddFaceState.waiting_for_name_selection)
async def name_selection_handler(message: Message, state: FSMContext):
    """Обработка выбора имени для добавления эмбеддингов."""
    selected_name = message.text.strip()

    if not selected_name:
        await message.answer("Имя не может быть пустым. Попробуйте снова.")
        return

    # Проверяем, что выбранное имя есть в базе данных
    if not await rq.check_name_exists(selected_name):
        await message.answer("Имя не найдено в базе данных. Попробуйте снова.")
        return

    # Сохраняем выбранное имя в состоянии
    await state.update_data(selected_name=selected_name)
    await message.answer(f"Вы выбрали имя: {selected_name}. Теперь отправьте фото для добавления эмбеддингов.")
    await state.set_state(AddFaceState.waiting_for_new_photo)

# Работало, начал добавлять новую базу
# @router.message(AddFaceState.waiting_for_name_selection)
# async def name_selection_handler(message: Message, state: FSMContext):
#     """Обработка выбора имени для добавления эмбеддингов"""
#     selected_name = message.text.strip()
#
#     # Проверяем, что выбранное имя есть в базе данных
#     if not await rq.check_name_exists(selected_name):
#         await message.answer("Имя не найдено в базе данных. Попробуйте снова.")
#         return
#
#     # Сохраняем выбранное имя в состоянии
#     await state.update_data(selected_name=selected_name)
#     await message.answer(f"Вы выбрали имя: {selected_name}. Теперь отправьте фото для добавления эмбеддингов.")
#     await state.set_state(AddFaceState.waiting_for_new_photo)

@router.message(AddFaceState.waiting_for_new_photo, lambda message: message.photo)
async def add_new_embedding_photo_handler(message: Message, state: FSMContext, bot: Bot):
    """Обработка новой фотографии для добавления эмбеддинга."""
    try:
        # Получаем данные из состояния (выбранное имя)
        data = await state.get_data()
        selected_name = data.get("selected_name")
        tg_id = message.from_user.id

        if not selected_name:
            await message.answer("Ошибка: имя не найдено в состоянии. Попробуйте снова.")
            await state.clear()
            return

        # Скачиваем фото
        photo = message.photo[-1]
        download_path = await download_photo(bot, photo.file_id, "add_")

        # Проверяем фото на наличие одного лица
        if not await validate_photo(download_path):
            await message.answer("Проблемы с фото. Проверьте требования и попробуйте снова.")
            os.remove(download_path)
            await state.clear()
            return

        # Сохраняем эмбеддинг в базу данных
        embed = await fc.save_embedding(download_path, selected_name, tg_id)
        if embed is not None:
            await message.answer(f"✅ Новый эмбеддинг для {selected_name} успешно добавлен!")
        else:
            await message.answer("❌ Ошибка при добавлении эмбеддинга.")

        # Удаляем временный файл
        os.remove(download_path)

        # Очищаем состояние
        await state.clear()

    except Exception as e:
        logger.error(f"Ошибка добавления эмбеддинга: {str(e)}", exc_info=True)
        await message.answer("Произошла ошибка при обработке фото.")
        await state.clear()


# Работало. начал новую базу
# # Обработчик, который будет обрабатывать фотографии, отправленные после выбора имени
# @router.message(AddFaceState.waiting_for_new_photo, lambda message: message.photo)
# async def add_new_embedding_photo_handler(message: Message, state: FSMContext, bot: Bot):
#     """Обработка новой фотографии для добавления эмбеддинга"""
#     try:
#         # Получаем данные из состояния (выбранное имя)
#         data = await state.get_data()
#         selected_name = data.get("selected_name")
#         tg_id = message.from_user.id
#
#         # Скачиваем фото
#         photo = message.photo[-1]
#         download_path = await download_photo(bot, photo.file_id, "add_")
#
#         # Проверяем фото на наличие одного лица
#         if not await validate_photo(download_path):
#             await message.answer("Проблемы с фото. Проверьте требования и попробуйте снова.")
#             os.remove(download_path)
#             await state.clear()
#             return
#
#         # Сохраняем эмбеддинг в базу данных
#         embed = await fc.save_embedding(download_path, selected_name, tg_id)
#         if embed is not None:
#             await message.answer(f"✅ Новый эмбеддинг для {selected_name} успешно добавлен!")
#         else:
#             await message.answer("❌ Ошибка при добавлении эмбеддинга.")
#
#         # Удаляем временный файл
#         os.remove(download_path)
#
#         # Очищаем состояние
#         await state.clear()
#
#     except Exception as e:
#         logger.error(f"Ошибка добавления эмбеддинга: {str(e)}")
#         await message.answer("Произошла ошибка при обработке фото.")
#         await state.clear()


# Обновим обработчик добавления лиц
@router.message(lambda message: message.text == "Добавить лицо")
async def add_face_start_handler(message: Message, state: FSMContext):
    """Начало процесса добавления лица"""
    await message.answer("Отправьте фото человека для добавления в базу")
    await state.set_state(MainState.adding_face.waiting_for_photo)

#  Новый роутер на добавление нескольких фото человека
@router.message(AddFaceState.waiting_for_photo, lambda message: message.photo)
async def add_face_photo_handler(message: Message, state: FSMContext, bot: Bot):
    try:
        photo = message.photo[-1]
        download_path = await download_photo(bot, photo.file_id, "add_")

        if not await validate_photo(download_path):
            await message.answer("Проблемы с фото. Проверьте требования и попробуйте снова.")
            os.remove(download_path)
            await state.clear()
            return

        await state.update_data(photo_path=download_path)
        await message.answer("Введите имя человека:")
        await state.set_state(AddFaceState.waiting_for_name)

    except Exception as e:
        logger.error(f"Ошибка добавления лица: {str(e)}")
        await message.answer("Произошла ошибка при обработке фото")
        await state.clear()

# # Старый роутер на добавление 1 фото человека
# @router.message(AddFaceState.waiting_for_photo, lambda message: message.photo)
# async def add_face_photo_handler(message: Message, state: FSMContext, bot: Bot):
#     """Обработка фото для добавления"""
#     try:
#         photo = message.photo[-1]
#         download_path = await download_photo(bot, photo.file_id, "add_")
#
#         if not await validate_photo(download_path):
#             await message.answer("Проблемы с фото. Проверьте требования и попробуйте снова.")
#             os.remove(download_path)
#             await state.clear()
#             return
#
#         await state.update_data(photo_path=download_path)
#         await message.answer("Введите имя человека:")
#         await state.set_state(AddFaceState.waiting_for_name)
#
#     except Exception as e:
#         logger.error(f"Ошибка добавления лица: {str(e)}")
#         await message.answer("Произошла ошибка при обработке фото")
#         await state.clear()


@router.message(AddFaceState.waiting_for_name)
async def add_face_name_handler(message: Message, state: FSMContext):
    """Обработка ввода имени с проверкой уникальности."""
    name = message.text.strip()
    tg_id = message.from_user.id

    # Проверка существования имени у других пользователей
    async with async_session() as session:
        existing = await session.execute(
            select(User).where(User.name == name)
        )
        if existing.scalars().first():
            await message.answer("⚠️ Это имя уже используется другим пользователем!")
            return

    if not name.replace(' ', '').isalnum():
        await message.answer("Имя должно содержать только буквы, цифры и пробелы")
        return

    await state.update_data(name=name, tg_id=tg_id)
    await message.answer(
        f"Добавить нового человека?\nИмя: {name}",
        reply_markup=kb.get_confirmation_keyboard()
    )
    await state.set_state(AddFaceState.confirmation)


@router.callback_query(AddFaceState.confirmation)
async def confirmation_handler(callback: CallbackQuery, state: FSMContext):
    """Обработка подтверждения."""
    data = await state.get_data()
    photo_path = data.get('photo_path')
    name = data.get('name')
    tg_id = data.get('tg_id')

    try:
        if callback.data == "confirm_add":
            # Сохраняем эмбеддинг
            embed = await fc.save_embedding(photo_path, name, tg_id)
            if embed is not None:
                await callback.message.answer("✅ Человек успешно добавлен в базу!")
            else:
                await callback.message.answer("❌ Ошибка при добавлении эмбеддинга.")
        else:
            await callback.message.answer("❌ Добавление отменено")

    except Exception as e:
        logger.error(f"Ошибка сохранения: {str(e)}")
        await callback.message.answer("❌ Ошибка при сохранении данных")

    finally:
        if photo_path and os.path.exists(photo_path):
            os.remove(photo_path)
        await state.clear()
        await callback.answer()


@router.message(lambda message: message.text == "Отмена")
async def cancel_handler(message: Message, state: FSMContext):
    """Обработчик отмены операций"""
    data = await state.get_data()

    if photo_path := data.get('photo_path'):
        if os.path.exists(photo_path):
            os.remove(photo_path)

    await state.clear()
    await message.answer("Операция отменена", reply_markup=kb.get_start_keyboard())

@router.message(F.text.lower() == "найди")
async def scan_photos_handler(message: Message):
    await message.answer("⏳ Начало обработки фотографий...")
    new_faces = await rq.process_directory()
    await message.answer(f"✅ Обработка завершена! Найдено новых лиц: {new_faces}")

# @router.message(F.text.lower() == "найди")
# async def scan_photos_handler(message: Message):
#     """Запуск обработки фотографий в папке"""
#     await message.answer("Начало обработки фотографий...")
#     await rq.process_directory()
#     await message.answer("Обработка завершена! Найдено новых фото: ...")

# #___________________________________________________________________________________________________________________
# #   Отправка всех найденных фото текстом
# #___________________________________________________________________________________________________________________
#
# @router.message(F.text.startswith("Найти "))
# async def find_photos_handler(message: Message):
#     """Обработчик поиска фотографий по имени"""
#     try:
#         name = message.text.split(" ", 1)[1].strip()
#         photos = await get_photos_by_name(name)
#
#         if not photos:
#             await message.answer(f"Фотографии с {name} не найдены.")
#             return
#
#         # Отправляем первые 10 результатов
#         response = f"Найдено {len(photos)} фото:\n" + "\n".join(photos[:14])
#         await message.answer(response)
#
#     except Exception as e:
#         logger.error(f"Ошибка поиска: {str(e)}")
#         await message.answer("Произошла ошибка при поиске.")
# #___________________________________________________________________________________________________________________
# #   Отправка всех найденных фото текстом
# #___________________________________________________________________________________________________________________

#___________________________________________________________________________________________________________________
#   Отправка всех найденных фото альбомами 10 штук в альбоме
#___________________________________________________________________________________________________________________

@router.message(F.text.startswith("Найти "))
async def find_photos_handler(message: Message):
    """Обработчик поиска с отправкой альбомами"""
    try:
        name = message.text.split(" ", 1)[1].strip()
        photos = await rq.get_photos_by_name(name)

        if not photos:
            await message.answer(f"Фотографии с именем {name} не найдены.")
            return

        base_path = "./user_photos/"
        total = len(photos)
        await message.answer(f"📁 Найдено {total} фото. Формирую альбомы...")

        success = 0
        errors = 0
        chunk_size = 10

        # Разбиваем фото на группы по 10
        for i in range(0, total, chunk_size):
            chunk = photos[i:i + chunk_size]
            media_group = []

            # Формируем альбом
            for photo_path in chunk:
                try:
                    full_path = os.path.join(base_path, photo_path)

                    if not os.path.exists(full_path):
                        logger.warning(f"Файл не найден: {full_path}")
                        errors += 1
                        continue

                    media_group.append(InputMediaPhoto(
                        media=FSInputFile(full_path),
                        caption=f"Фото {i + 1}-{i + len(chunk)}" if len(media_group) == 0 else None
                    ))
                    success += 1

                except Exception as e:
                    logger.error(f"Ошибка подготовки {photo_path}: {str(e)}")
                    errors += 1

            # Отправляем альбом если есть фото
            if media_group:
                try:
                    await message.answer_media_group(media_group)
                    await asyncio.sleep(1)  # Задержка между альбомами

                    # Прогресс каждые 50 файлов
                    if (i // chunk_size) % 5 == 0:
                        await message.answer(f"🚀 Отправлено {min(i + chunk_size, total)}/{total}")

                except Exception as e:
                    logger.error(f"Ошибка отправки альбома: {str(e)}")
                    errors += len(media_group)
                    success -= len(media_group)

        # Итоговое сообщение
        await message.answer(
            f"✅ Все альбомы отправлены!\n"
            f"✅ Успешно: {success} фото\n"
            f"❌ Пропущено: {errors}"
        )

    except Exception as e:
        logger.error(f"Ошибка поиска: {str(e)}", exc_info=True)
        await message.answer("❌ Произошла критическая ошибка при обработке запроса.")

#___________________________________________________________________________________________________________________
#   Отправка всех найденных фото альбомами 10 штук в альбоме
#___________________________________________________________________________________________________________________
#
# #___________________________________________________________________________________________________________________
# #   Отправка всех найденных фото
# #___________________________________________________________________________________________________________________
#
# @router.message(F.text.startswith("Найти "))
# async def find_photos_handler(message: Message):
#     """Обработчик поиска фотографий по имени"""
#     try:
#         name = message.text.split(" ", 1)[1].strip()
#         photos = await get_photos_by_name(name)
#
#         if not photos:
#             await message.answer(f"Фотографии с именем {name} не найдены.")
#             return
#
#         # Указываем базовый путь к папке с фото
#         base_path = "./"
#
#         # Отправляем уведомление о начале отправки
#         total = len(photos)
#         await message.answer(f"🔍 Найдено {total} фото. Начинаю отправку...")
#
#         # Отправляем все фото
#         success = 0
#         errors = 0
#
#         for idx, photo_path in enumerate(photos, 1):  # Убрали срез [:10]
#             try:
#                 full_path = os.path.join(base_path, photo_path)
#
#                 if not os.path.exists(full_path):
#                     logger.warning(f"Файл не найден: {full_path}")
#                     errors += 1
#                     continue
#
#                 file = FSInputFile(full_path)
#                 await message.answer_document(file)
#                 success += 1
#
#                 # Обновляем статус каждые 10 файлов
#                 if idx % 10 == 0:
#                     await message.answer(f"📤 Отправлено {idx}/{total}...")
#
#                 # Увеличиваем задержку для надежности
#                 await asyncio.sleep(1)  # Было 0.5
#
#             except Exception as e:
#                 logger.error(f"Ошибка отправки {photo_path}: {str(e)}")
#                 errors += 1
#
#         # Финал отправки
#         await message.answer(
#             f"✅ Готово! Успешно отправлено: {success}\n"
#             f"❌ Ошибок: {errors}"
#         )
#
#     except Exception as e:
#         logger.error(f"Ошибка поиска: {str(e)}", exc_info=True)
#         await message.answer("Произошла критическая ошибка при поиске.")
# #___________________________________________________________________________________________________________________
# #   Отправка всех найденных фото
# #___________________________________________________________________________________________________________________


# #___________________________________________________________________________________________________________________
# #   Отправка 10 первых найденных фото
# #___________________________________________________________________________________________________________________
#
# @router.message(F.text.startswith("Найти "))
# async def find_photos_handler(message: Message):
#     """Обработчик поиска фотографий по имени"""
#     try:
#         name = message.text.split(" ", 1)[1].strip()
#         photos = await get_photos_by_name(name)
#
#         if not photos:
#             await message.answer(f"Фотографии с именем {name} не найдены.")
#             return
#
#         # Указываем базовый путь к папке с фото (если нужно)
#         base_path = "./"
#
#         # Отправляем первые 10 результатов
#         await message.answer(f"Найдено {len(photos)} фото. Отправляю первые 10:")
#
#         for photo_path in photos[:10]:
#             try:
#                 # Собираем полный путь к файлу
#                 full_path = os.path.join(base_path, photo_path)
#
#                 # Проверяем существование файла
#                 if not os.path.exists(full_path):
#                     logger.warning(f"Файл не найден: {full_path}")
#                     continue
#
#                 # Отправляем файл
#                 file = FSInputFile(full_path)
#                 await message.answer_document(file)
#
#                 # Небольшая задержка для избежания лимитов
#                 await asyncio.sleep(0.5)
#
#             except Exception as e:
#                 logger.error(f"Ошибка отправки файла {photo_path}: {str(e)}")
#
#     except Exception as e:
#         logger.error(f"Ошибка поиска: {str(e)}", exc_info=True)
#         await message.answer("Произошла ошибка при поиске.")
# #___________________________________________________________________________________________________________________
# #   Отправка 10 первых найденных фото
# #___________________________________________________________________________________________________________________
