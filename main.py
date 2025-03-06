import os
import asyncio
import logging

import faiss
import numpy as np
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types


from app.database.models import async_main
from app.handlers import router
from app.database.requests import load_embeddings
from bot_cmd_list import private
from config import update_known_embeddings, face_embeds_index, FACE_EMBED_INDEX_PATH, known_embeddings_names


async def main():
# Запуск функции базы данных в самом начале, для того что-бы при запуске бота создавались все таблицы.
    await  async_main()

# Загрузка переменных окружения
    load_dotenv()

    loaded_embeddings = await load_embeddings()
    # Обновляем глобальную переменную
    update_known_embeddings(loaded_embeddings)


# # Загружаем эмбеддинги при старте
#     global known_embeddings
#     known_embeddings = await load_embeddings()

# Инициализация бота и диспетчера
    bot = Bot(token=os.getenv("BOT_TOKEN")) #, parse_mode=ParseMode.HTML)
    dp = Dispatcher() #Основной роутер обрабатывает входящие обновления, сообщения, calback

    # Регистрация обработчика завершения
    dp.shutdown.register(on_shutdown)

    #Вызываем метод include_router
    dp.include_router(router) # этот роутер сработает первым
    # dp.include_router(gr_router) # этот роутер сработает если первый не сработал
    # Создаём кнопку меню с командами
    await  bot.set_my_commands(commands=private, scope=types.BotCommandScopeAllPrivateChats())
    # # Удаление кнопок
    # await bot.delete_my_commands(scope=types.BotCommandScopeAllPrivateChats())
    await dp.start_polling(bot) #start_polling хэндлеры



# обработчик для сохранения индекса перед выходом
async def on_shutdown():
    """Сохранение индексов FAISS при завершении работы."""
    try:
        faiss.write_index(face_embeds_index, FACE_EMBED_INDEX_PATH)
        np.save("face_names.npy", known_embeddings_names)
        logging.info("Индексы FAISS сохранены.")
    except Exception as e:
        logging.error(f"Ошибка при сохранении индексов: {str(e)}")

# конструкция, которая запускает функцию Main
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Принудительное сохранение индексов
        asyncio.run(on_shutdown())
        print('ВыключилиБота')