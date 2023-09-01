'''
===========================================
        Module: Open-source LLM Setup
===========================================
'''
import argparse
import box
from dotenv import find_dotenv, load_dotenv
from langchain.llms import LlamaCpp
from multipledispatch import dispatch
import timeit
from src.utils import setup_dbqa
import yaml

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

@dispatch()
def build_llm():
    llm = LlamaCpp(
        model_path='models\llama-2-13b-chat.gguf.q8_0.bin',
        n_ctx= 2048,
        verbose=True,
        use_mlock=True,
        n_gpu_layers=100,
        n_threads=4,
        n_batch=256,
        temperature=cfg.TEMPERATURE,
        max_tokens=cfg.MAX_NEW_TOKENS
    )
    return llm

@dispatch(str)
def query(qstring: str):
    start = timeit.default_timer()
    dbqa = setup_dbqa()
    response = dbqa({'query': qstring})
    end = timeit.default_timer()

    print('='*100)
    print(f"query: {response['query']}")
    print(f"result: {response['result']}")
    print(f"time: {round(end - start, 2)} seconds")
    print('='*100)

    return response

@dispatch(str, str)
def query(qstring: str, db_path: str):
    start = timeit.default_timer()
    dbqa = setup_dbqa(db_path)
    response = dbqa({'query': qstring})
    end = timeit.default_timer()
    response['time'] = round(end - start, 2)

    print('='*100)
    print(f"query: {response['query']}")
    print(f"result: {response['result']}")
    print(f"time: {response['time']} seconds")
    print('='*100)

    return response

@dispatch(str, str, LlamaCpp)
def query(qstring: str, db_path: str, llm):
    start = timeit.default_timer()
    dbqa = setup_dbqa(db_path, llm)
    response = dbqa({'query': qstring})
    end = timeit.default_timer()
    response['time'] = round(end - start, 2)

    print('='*100)
    print(f"query: {response['query']}")
    print(f"result: {response['result']}")
    print(f"time: {response['time']} seconds")
    print('='*100)

    return response

def log_response(response):
    # Process source documents
    source_docs = response['source_documents']
    for i, doc in enumerate(source_docs):
        print(f'Relevant text sample #{i+1} from page #{doc.metadata["page"]}:')
        print(f'{doc.page_content}')
        # print(f'Document Name: {doc.metadata["source"]}')
        print('='* 100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        type=str,
                        default='How much is the minimum guarantee payable by adidas?',
                        help='Enter the query to pass into the LLM')
    args = parser.parse_args()

    response = query(args.input)
    log_response(response)