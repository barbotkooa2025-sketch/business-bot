import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

# Новые импорты для LangChain 1.x
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain

import chromadb

load_dotenv()

def setup_vectorstore():
    print("📂 Загрузка готовой базы знаний из chroma_db...")
    
    if not os.path.exists("./chroma_db"):
        raise FileNotFoundError("Папка chroma_db не найдена!")
    
    embeddings = CohereEmbeddings(
        model="embed-multilingual-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )
    
    client = chromadb.PersistentClient(path="./chroma_db")
    
    vectorstore = Chroma(
        client=client,
        collection_name="knowledge_base",
        embedding_function=embeddings
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
    prompt = PromptTemplate.from_template(template)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)
