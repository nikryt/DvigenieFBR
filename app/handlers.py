import os
import cv2
import logging
from pathlib import Path
from aiogram import Router, Bot, F
from aiogram.types import Message, CallbackQuery
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup


from app.database.requests import add_embedding, process_directory


from app.keyboards import get_start_keyboard, get_confirmation_keyboard
from app.recognition.face import recognize_face, save_embedding, mtcnn, get_photos_by_name
from config import known_embeddings



router = Router()
logger = logging.getLogger(__name__)

# Создаем необходимые директории
Path("downloads").mkdir(exist_ok=True)
Path("app/recognition/known_faces").mkdir(parents=True, exist_ok=True)


class AddFaceState(StatesGroup):
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
        faces = mtcnn(image)

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
        reply_markup=get_start_keyboard(),
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

        if result := await recognize_face(download_path):
            await message.answer(f"На фотографии есть: {result}")
        else:
            await message.answer("Не удалось распознать человека.")

        os.remove(download_path)
        await state.clear()

    except Exception as e:
        logger.error(f"Recognition error: {str(e)}")
        await message.answer("Ошибка распознавания. Попробуйте еще раз.")
        await state.clear()


# Обновим обработчик добавления лиц
@router.message(lambda message: message.text == "Добавить лицо")
async def add_face_start_handler(message: Message, state: FSMContext):
    """Начало процесса добавления лица"""
    await message.answer("Отправьте фото человека для добавления в базу")
    await state.set_state(MainState.adding_face.waiting_for_photo)


@router.message(AddFaceState.waiting_for_photo, lambda message: message.photo)
async def add_face_photo_handler(message: Message, state: FSMContext, bot: Bot):
    """Обработка фото для добавления"""
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


@router.message(AddFaceState.waiting_for_name)
async def add_face_name_handler(message: Message, state: FSMContext):
    """Обработка ввода имени"""
    name = message.text.strip()
    tg_id = message.from_user.id
    if not name.replace(' ', '').isalnum():
        await message.answer("Имя должно содержать только буквы, цифры и пробелы")
        return

    await state.update_data(name=name, tg_id=tg_id)
    await message.answer(
        f"Добавить нового человека?\nИмя: {name}",
        reply_markup=get_confirmation_keyboard()
    )
    await state.set_state(AddFaceState.confirmation)


@router.callback_query(AddFaceState.confirmation)
async def confirmation_handler(callback: CallbackQuery, state: FSMContext):
    """Обработка подтверждения"""
    data = await state.get_data()
    photo_path = data.get('photo_path')

    try:
        if callback.data == "confirm_add":
            embed = await save_embedding(data['photo_path'], data['name'], data['tg_id'])  # Добавляем data['name']
            await add_embedding(name=data['name'], tg_id=data['tg_id'], embedding=embed)
            await callback.message.answer("✅ Человек успешно добавлен в базу!")
            # Обновляем кеш эмбеддингов
            known_embeddings[data['name']] = embed  # Добавляем в словарь
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
    await message.answer("Операция отменена", reply_markup=get_start_keyboard())

@router.message(F.text.lower() == "найди")
async def scan_photos_handler(message: Message):
    """Запуск обработки фотографий в папке"""
    await message.answer("Начало обработки фотографий...")
    await process_directory()
    await message.answer("Обработка завершена! Найдено новых фото: ...")


@router.message(F.text.startswith("Найти "))
async def find_photos_handler(message: Message):
    """Обработчик поиска фотографий по имени"""
    try:
        name = message.text.split(" ", 1)[1].strip()
        photos = await get_photos_by_name(name)

        if not photos:
            await message.answer(f"Фотографии с {name} не найдены.")
            return

        # Отправляем первые 10 результатов
        response = f"Найдено {len(photos)} фото:\n" + "\n".join(photos[:14])
        await message.answer(response)

    except Exception as e:
        logger.error(f"Ошибка поиска: {str(e)}")
        await message.answer("Произошла ошибка при поиске.")