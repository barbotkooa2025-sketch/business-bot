import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv()

# Используем вашу родную папку!
PERSIST_DIRECTORY = "./knowledge_base_chroma" 

def setup_vectorstore():
    print("📂 Инициализация базы знаний...")
    
    embeddings = CohereEmbeddings(
        model="embed-multilingual-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )
    
    # Если база уже создана - просто загружаем её
    if os.path.exists(PERSIST_DIRECTORY):
        print("♻️ База уже существует, загружаю...")
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
            collection_name="knowledge_base"
        )
    else:
        # Если базы нет - создаем её из ваших файлов
        print("🛠 Базы нет. Запускаю create_knowledge_base.py...")
        os.system("python create_knowledge_base.py")
        
        print("✅ База создана, загружаю...")
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
            collection_name="knowledge_base"
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
