import os
import glob
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import chromadb

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
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_documents)
    print(f"Документы разбиты на {len(texts)} фрагментов")
    
    # Используем встроенные эмбеддинги ChromaDB (работают локально!)
    print("Инициализация встроенных эмбеддингов ChromaDB...")
    client_settings = chromadb.config.Settings(
        is_persistent=True,
        persist_directory="./chroma_db",
        anonymized_telemetry=False
    )
    
    # ChromaDB использует встроенную модель эмбеддингов по умолчанию
    vectorstore = Chroma.from_documents(
        documents=texts,
        collection_name="knowledge_base",
        persist_directory="./chroma_db",
        client_settings=client_settings
    )
    print("✅ База знаний готова!")
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
