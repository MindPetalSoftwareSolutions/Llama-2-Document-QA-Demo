# =========================
#  Module: Vector DB Build
# =========================
import box
import yaml
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from multipledispatch import dispatch
from torch import cuda
from src.env import device

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


# Build vector database
@dispatch(str)
def run_db_build(glob:str = '*.pdf'):
    loader = DirectoryLoader(cfg.DATA_PATH,
                             glob=glob,
                             loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.CHUNK_SIZE,
                                                   chunk_overlap=cfg.CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': device})

    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(cfg.DB_FAISS_PATH)

@dispatch(str, str, str, chunk_size=int, chunk_overlap=int)
def run_db_build(filename, data_path, db_faiss_path, chunk_size=None, chunk_overlap=None):
    chunk_size = chunk_size or cfg.CHUNK_SIZE
    chunk_overlap = chunk_overlap or cfg.CHUNK_OVERLAP
    
    loader = DirectoryLoader(data_path,
                             glob=filename,
                             loader_cls=PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': device})

    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(db_faiss_path)

if __name__ == "__main__":
    run_db_build()