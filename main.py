import box
import timeit
import yaml
import argparse
from dotenv import find_dotenv, load_dotenv
from src.utils import setup_dbqa

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

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