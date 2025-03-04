import os
import asyncio
import logging

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.enums import ParseMode

from app.database.models import async_main
from app.handlers import router
from app.database.requests import load_embeddings
from bot_cmd_list import private
from config import known_embeddings

async def main():
# Запуск функции базы данных в самом начале, для того что-бы при запуске бота создавались все таблицы.
    await  async_main()

# Загрузка переменных окружения
    load_dotenv()

# Загружаем эмбеддинги при старте
    global known_embeddings
    known_embeddings = await load_embeddings()

# Инициализация бота и диспетчера
    bot = Bot(token=os.getenv("BOT_TOKEN")) #, parse_mode=ParseMode.HTML)
    dp = Dispatcher() #Основной роутер обрабатывает входящие обновления, сообщения, calback
    #Вызываем метод include_router
    dp.include_router(router) # этот роутер сработает первым
    # dp.include_router(gr_router) # этот роутер сработает если первый не сработал
    # Создаём кнопку меню с командами
    await  bot.set_my_commands(commands=private, scope=types.BotCommandScopeAllPrivateChats())
    # # Удаление кнопок
    # await bot.delete_my_commands(scope=types.BotCommandScopeAllPrivateChats())
    await dp.start_polling(bot) #start_polling хэндлеры

# конструкция, которая запускает функцию Main
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('ВыключилиБота')