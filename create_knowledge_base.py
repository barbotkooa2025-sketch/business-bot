import os
import glob
import time
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import CohereEmbeddings
import chromadb
from chromadb.config import Settings

load_dotenv()

def create_knowledge_base():
    print("🚀 Начинаем создание базы знаний...")
    
    txt_files = glob.glob('./knowledge_base/*.txt')
    
    if not txt_files:
        print("❌ В папке knowledge_base не найдено ни одного .txt файла!")
        return
    
    print(f"Найдено файлов: {len(txt_files)}")
    
    all_documents = []
    for file_path in txt_files:
        for encoding in ['utf-8', 'cp1251', 'latin-1']:
            try:
                loader = TextLoader(file_path, encoding=encoding)
                docs = loader.load()
                all_documents.extend(docs)
                print(f"✅ Загружен ({encoding}): {file_path}")
                break
            except Exception as e:
                continue
    
    if not all_documents:
        print("❌ Не удалось загрузить документы!")
        return
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(all_documents)
    print(f"Документы разбиты на {len(texts)} фрагментов")
    
    print("Инициализация эмбеддингов через Cohere API...")
    embeddings = CohereEmbeddings(
        model="embed-multilingual-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )
    
    client = chromadb.Client(Settings(
        is_persistent=True,
        persist_directory="./chroma_db",
        anonymized_telemetry=False
    ))
    
    collection = client.get_or_create_collection(
        name="knowledge_base",
        metadata={"hnsw:space": "cosine"}
    )
    
    batch_size = 50
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        print(f"⏳ Обработка батча {batch_num}/{total_batches} ({len(batch)} фрагментов)...")
        
        batch_texts = [doc.page_content for doc in batch]
        batch_metadatas = [{"source": doc.metadata.get("source", "unknown")} for doc in batch]
        batch_ids = [f"doc_{i+j}" for j in range(len(batch))]
        
        try:
            batch_embeddings = embeddings.embed_documents(batch_texts)
            
            collection.add(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )
            
            if batch_num < total_batches:
                print("⏸️ Пауза 5 секунд...")
                time.sleep(5)
                
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            print("⏸️ Пауза 60 секунд...")
            time.sleep(60)
            continue
    
    print(f"✅ База знаний готова! Добавлено {len(texts)} фрагментов.")
    print("📁 Данные сохранены в папке ./chroma_db")

if __name__ == "__main__":
    create_knowledge_base()
