import os
import zipfile
import urllib.request
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv()

# ⚠️ ЗАМЕНИТЕ ЭТУ ССЫЛКУ НА СВОЮ ИЗ ШАГА 3!
CHROMA_ZIP_URL = "https://github.com/barbotkooa2025-sketch/business-bot/releases/download/v1.0.0/chroma_db.zip"
CHROMA_DIR = "./chroma_db"
CHROMA_ZIP = "./chroma_db.zip"

def download_and_extract():
    """Скачивает и распаковывает базу знаний."""
    print(" Скачиваю базу знаний...")
    urllib.request.urlretrieve(CHROMA_ZIP_URL, CHROMA_ZIP)
    
    print("📦 Распаковываю...")
    with zipfile.ZipFile(CHROMA_ZIP, 'r') as zip_ref:
        zip_ref.extractall(".")
    
    os.remove(CHROMA_ZIP)
    print("✅ База знаний готова!")

def setup_vectorstore():
    print("📂 Загрузка базы знаний...")
    
    # Если папки нет — скачиваем и распаковываем
    if not os.path.exists(CHROMA_DIR):
        download_and_extract()
    
    embeddings = CohereEmbeddings(
        model="embed-multilingual-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )
    
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="knowledge_base"
    )
    
    print("✅ База знаний загружена!")
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
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    return chain
