import os
import glob
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import chromadb
from chromadb.config import Settings

load_dotenv()

def setup_vectorstore():
    txt_files = glob.glob('./knowledge_base/*.txt')
    
    if not txt_files:
        raise FileNotFoundError("В папке knowledge_base не найдено ни одного .txt файла!")
    
    print(f"Найдено файлов для загрузки: {len(txt_files)}")
    
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
        else:
            print(f"❌ Не удалось загрузить {file_path}")
    
    if not all_documents:
        raise ValueError("Не удалось загрузить ни одного документа!")
    
    # Увеличим размер чанка, чтобы уменьшить их количество
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # Было 1000
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(all_documents)
    print(f"Документы разбиты на {len(texts)} фрагментов")
    
    # Инициализация эмбеддингов Cohere
    print("Инициализация эмбеддингов через Cohere API...")
    embeddings = CohereEmbeddings(
        model="embed-multilingual-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )
    
    # Создаём Chroma клиент вручную для контроля батчинга
    client = chromadb.Client(Settings(
        is_persistent=True,
        persist_directory="./chroma_db",
        anonymized_telemetry=False
    ))
    
    # Создаём или получаем коллекцию
    collection = client.get_or_create_collection(
        name="knowledge_base",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Добавляем тексты батчами по 96 штук (лимит Cohere)
    batch_size = 96
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        print(f"Обработка батча {batch_num}/{total_batches} ({len(batch)} фрагментов)...")
        
        # Получаем тексты и метаданные
        batch_texts = [doc.page_content for doc in batch]
        batch_metadatas = [{"source": doc.metadata.get("source", "unknown")} for doc in batch]
        batch_ids = [f"doc_{i+j}" for j in range(len(batch))]
        
        # Получаем эмбеддинги через Cohere
        batch_embeddings = embeddings.embed_documents(batch_texts)
        
        # Добавляем в Chroma
        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas
        )
    
    print(f"✅ База знаний готова! Добавлено {len(texts)} фрагментов.")
    
    # Возвращаем Chroma как vectorstore для LangChain
    vectorstore = Chroma(
        client=client,
        collection_name="knowledge_base",
        embedding_function=embeddings
    )
    
    return vectorstore

def get_chain(vectorstore):
    llm = HuggingFaceEndpoint(
        repo_id=os.getenv("HF_MODEL_ID"),
        huggingfacehub_api_token=os.getenv("HF_TOKEN"),
        temperature=0.3,
        max_new_tokens=1024
    )

    template = """Ты — эксперт по еврейскому менеджменту, бизнес-процессам и философии. 
    Ты отвечаешь на вопросы, опираясь на принципы Торы, Талмуда, Шульхан Арух (особенно раздел Хошен Мишпат) 
    и современную еврейскую бизнес-этику. 
    Если информации в контексте недостаточно, честно скажи об этом, но постарайся дать общий этический совет.
    
    Контекст из базы знаний:
    {context}

    Вопрос пользователя:
    {question}

    Ответ (структурированно, с уважением и ссылками на источники):
    """
    prompt = PromptTemplate.from_template(template)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)
