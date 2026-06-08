import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
from rag import setup_vectorstore, get_chain

load_dotenv()

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Инициализация RAG при запуске
print("Загрузка базы знаний и инициализация модели...")
vectorstore = setup_vectorstore()
chain = get_chain(vectorstore)
print("Бот готов к работе!")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Шалом! Я ваш помощник по вопросам еврейского менеджмента, бизнес-этики и философии.\n"
        "Задайте мне вопрос о деловых процессах, контрактах или жизненных ситуациях."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_question = update.message.text
    await update.message.reply_text("Обрабатываю ваш вопрос с точки зрения еврейской мудрости... ⏳")
    
    try:
        # Вызов LLM через цепочку RAG
        result = chain.invoke({"input": user_question})
        answer = result["answer"]
        await update.message.reply_text(answer)
    except Exception as e:
        logging.error(f"Ошибка при запросе к LLM: {e}")
        await update.message.reply_text("Извините, произошла ошибка при обработке запроса. Попробуйте позже.")

def main():
    bot_token = os.getenv("BOT_TOKEN")
    application = Application.builder().token(bot_token).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Использование webhook для Render
    application.run_webhook(
        listen="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        url_path=bot_token,
        webhook_url=f"https://{os.environ.get('RENDER_EXTERNAL_HOSTNAME')}/{bot_token}" if os.environ.get("RENDER_EXTERNAL_HOSTNAME") else None
    )

if __name__ == '__main__':
    main()
