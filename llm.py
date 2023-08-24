'''
===========================================
        Module: Open-source LLM Setup
===========================================
'''
from langchain.llms import CTransformers
from dotenv import find_dotenv, load_dotenv
import box
import yaml
import timeit
import argparse
from src.utils import setup_dbqa
from multipledispatch import dispatch

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

@dispatch()
def build_llm():
    # Local CTransformers model
    llm = CTransformers(model=cfg.MODEL_BIN_PATH,
                        model_type=cfg.MODEL_TYPE,
                        config={'max_new_tokens': cfg.MAX_NEW_TOKENS,
                                'temperature': cfg.TEMPERATURE})
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