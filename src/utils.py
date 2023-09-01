'''
===========================================
        Module: Util functions
===========================================
'''
import box
import yaml

from src.env import device
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.vectorstores import FAISS
from multipledispatch import dispatch

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def set_prompt():
    """ Prompt template for QA retrieval for each vectorstore """
    template = """Use the following pieces of information to answer the user's question.

    Context: {context}
    Question: {question}

    Return the answer below and nothing else. The answer should be as brief as possible and omit unnecessary context. 
    The answer to the question is:
    """

    prompt = PromptTemplate(template=template,
                            input_variables=['context', 'question'])
    return prompt


def build_retrieval_qa(llm, prompt, vectordb):
    dbqa = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=vectordb.as_retriever(search_kwargs={'k': cfg.VECTOR_COUNT}),
                                       return_source_documents=cfg.RETURN_SOURCE_DOCUMENTS,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return dbqa

@dispatch(str, LlamaCpp)
def setup_dbqa(db_faiss_path, llm):

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': device})
    vectordb = FAISS.load_local(db_faiss_path, embeddings)
    qa_prompt = set_prompt()
    dbqa = build_retrieval_qa(llm, qa_prompt, vectordb)

    return dbqa