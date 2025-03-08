from aiogram.utils.keyboard import ReplyKeyboardBuilder, InlineKeyboardBuilder

def get_start_keyboard():
    builder = ReplyKeyboardBuilder()
    builder.button(text="Распознать лицо")
    builder.button(text="Добавить лицо")
    builder.button(text="Добавить эмбеддинги")
    builder.button(text="Отмена")
    builder.button(text="найди")
    builder.button(text="проверь")
    builder.button(text="Найти Никита")
    builder.adjust(2, 1, 1, 3)
    return builder.as_markup(resize_keyboard=True)

def get_confirmation_keyboard():
    builder = InlineKeyboardBuilder()
    builder.button(text="✅ Подтвердить", callback_data="confirm_add")
    builder.button(text="❌ Отменить", callback_data="cancel_add")
    return builder.as_markup()

def get_names_keyboard(names):
    """Создаёт клавиатуру с именами из базы данных."""
    builder = ReplyKeyboardBuilder()
    for name in names:
        builder.button(text=name)
    builder.button(text="Отмена")
    builder.adjust(2)
    return builder.as_markup(resize_keyboard=True)

def get_cancel_keyboard():
    builder = ReplyKeyboardBuilder()
    builder.button(text="Отмена")
    builder.adjust(1)
    return builder.as_markup(resize_keyboard=True)

def get_export_keyboard(names: list):
    """Клавиатура для выбора пользователя"""
    builder = ReplyKeyboardBuilder()
    for name in names:
        builder.button(text=f"🔹 {name}")
    builder.button(text="📋 Весь список")
    builder.button(text="❌ Отмена")
    builder.adjust(2)
    return builder.as_markup(resize_keyboard=True)

