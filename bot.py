import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
from rag import setup_vectorstore, get_chain

load_dotenv()

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

print("🚀 Запуск бота...")
print("📦 Загрузка базы знаний и инициализация модели...")
try:
    vectorstore = setup_vectorstore()
    chain = get_chain(vectorstore)
    print("✅ Бот готов к работе!")
except Exception as e:
    print(f"❌ Ошибка при инициализации: {e}")
    import traceback
    traceback.print_exc()
    raise

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Шалом! Я ваш помощник по вопросам еврейского менеджмента, бизнес-этики и философии.\n"
        "Задайте мне вопрос о деловых процессах, контрактах или жизненных ситуациях."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_question = update.message.text
    await update.message.reply_text("Обрабатываю ваш вопрос с точки зрения еврейской мудрости... ⏳")
    
    try:
        result = chain.invoke({"query": user_question})
        answer = result["result"]
        await update.message.reply_text(answer)
    except Exception as e:
        logger.error(f"Ошибка при запросе к LLM: {e}")
        await update.message.reply_text("Извините, произошла ошибка при обработке запроса. Попробуйте позже.")

def main():
    bot_token = os.getenv("BOT_TOKEN")
    if not bot_token:
        raise ValueError("BOT_TOKEN не установлен в переменных окружения!")
    
    application = Application.builder().token(bot_token).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    render_url = os.environ.get('RENDER_EXTERNAL_HOSTNAME')
    port = int(os.environ.get("PORT", 8080))
    
    if render_url:
        webhook_url = f"https://{render_url}/{bot_token}"
        print(f"🌐 Запускаю webhook: {webhook_url}")
        application.run_webhook(
            listen="0.0.0.0",
            port=port,
            url_path=bot_token,
            webhook_url=webhook_url
        )
    else:
        print("⚠️ RENDER_EXTERNAL_HOSTNAME не найден, запускаю polling")
        application.run_polling(drop_pending_updates=True)

if __name__ == '__main__':
    main()
