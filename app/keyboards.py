from aiogram.utils.keyboard import ReplyKeyboardBuilder, InlineKeyboardBuilder

def get_start_keyboard():
    builder = ReplyKeyboardBuilder()
    builder.button(text="Распознать лицо")
    builder.button(text="Добавить лицо")
    builder.button(text="Отмена")
    builder.adjust(2, 1)
    return builder.as_markup(resize_keyboard=True)

def get_confirmation_keyboard():
    builder = InlineKeyboardBuilder()
    builder.button(text="✅ Подтвердить", callback_data="confirm_add")
    builder.button(text="❌ Отменить", callback_data="cancel_add")
    return builder.as_markup()