import argparse
from crawler import crawl_llama_index_blog
from init_index import initializing_db, load_index
from utils.retry import retry_with_exponential_backoff
from query import get_query_engine
from llama_index.core.response.pprint_utils import pprint_response, pprint_metadata
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai.utils import ALL_AVAILABLE_MODELS, CHAT_MODELS
from llama_index.core import Settings

PERSIST_DB_DIR = "./db/db_storage/"
ALL_AVAILABLE_MODELS["gpt-4o-mini"]= 128000
CHAT_MODELS["gpt-4o-mini"] = 128000
docstore = SimpleDocumentStore()
index_store=SimpleIndexStore()
vector_store = SimpleVectorStore()

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini")

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--re-crawl", help="Re-crawl the blog before running queries.", action="store_true")
    parser.add_argument("--eval", help="Run evaluation on the test set.", action="store_true")
    args = parser.parse_args()

    if args.re_crawl:
        # Reinitialize the db and pipeline
        input("Enter to start crawl blog posts....")
        crawl_llama_index_blog()
        
        input("Enter to start re-index....")
        initializing_db(docstore=docstore, index_store=index_store, vector_store=vector_store, embedding_model=Settings.embed_model)
        
    print("Loading index from local storage....")
    index = load_index(docstore=docstore, index_store=index_store, vector_store=vector_store)
    
    query_engine = get_query_engine(index)
    # query_engine = index.as_query_engine()
    if args.eval:
        from eval import run_evaluation
        retry_support = input("Do you want to use retry with exponential backoff for generating answer and evaluation? (y/n): ")
        if retry_support.lower() == 'y':
            run_evaluation(query_engine, with_retry_support=True)
        else:
            run_evaluation(query_engine)
    
    @retry_with_exponential_backoff
    def ask_question(**kwargs):
        return query_engine.query(**kwargs)

    show_metadata = input("Do you want to see metadata with the response? (y/n): ")
    show_metadata = False if show_metadata.lower() == 'n' else True
    
    while True:
        query = input("\n\nEnter your query: ")
        if query == 'exit':
            break
        response = ask_question(str_or_query_bundle=query)
        print("\n**Response**")
        pprint_response(response=response, show_source=True, source_length=300)

        if show_metadata:
            print("\n**Metadata**")
            pprint_metadata(response.metadata)

if __name__ == "__main__":
    main()


