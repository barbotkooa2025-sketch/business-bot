import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# 1. Загрузка и векторизация базы знаний (выполняется один раз при старте)
def setup_vectorstore():
    loader = DirectoryLoader('./knowledge_base', glob="**/*.txt") # Добавьте *.pdf если нужно
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # Используем бесплатные эмбеддинги Hugging Face
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="./chroma_db")
    return vectorstore

# 2. Настройка LLM и промпта
def get_chain(vectorstore):
    llm = HuggingFaceEndpoint(
        repo_id=os.getenv("HF_MODEL_ID"),
        huggingfacehub_api_token=os.getenv("HF_TOKEN"),
        temperature=0.3, # Низкая температура для точности в юридических/этических вопросах
        max_new_tokens=1024
    )

    # Системный промпт, задающий роль бота
    template = """Ты — эксперт по еврейскому менеджменту, бизнес-процессам и философии. 
    Ты отвечаешь на вопросы, опираясь на принципы Торы, Талмуда, Шульхан Арух (особенно раздел Хошен Мишпат) 
    и современную еврейскую бизнес-этику. 
    Если вопрос касается контрактов, упоминай важность честности, избегания онаа (обмана) и соблюдения договоренностей.
    Если информации в контексте недостаточно, честно скажи об этом, но постарайся дать общий этический совет.
    
    Контекст из базы знаний:
    {context}

    Вопрос пользователя:
    {question}

    Ответ (структурированно, с уважением и ссылками на источники, если они есть в контексте):
    """
    prompt = PromptTemplate.from_template(template)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Берем 3 наиболее релевантных фрагмента
    
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)
