import os
import logging
import sys
from llama_index.core import (
    SummaryIndex,
    VectorStoreIndex,
    StorageContext, 
    MockEmbedding,
    load_index_from_storage
)
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.node_parser import SentenceSplitter, MarkdownNodeParser
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.embeddings.openai import OpenAIEmbedding
import utils.csv_processor as csv_processor

PERSIST_DB_DIR = "./db/db_storage/"
# option to try  text-embedding-ada-002	, text-embedding-3-large, text-embedding-3-small
embedding_model = OpenAIEmbedding(model="text-embedding-3-small")
# We can test with SentenceSplitter(chunk_size=1024, chunk_overlap=10), however MarkdownNodeParser works better since we have markdown data
chunker = MarkdownNodeParser() 

def initializing_db(
        overwriting_db=True, 
        docstore = SimpleDocumentStore(), 
        index_store=SimpleIndexStore(), 
        vector_store = SimpleVectorStore(), 
        embedding_model = OpenAIEmbedding(model="text-embedding-3-small"),
        chunker = MarkdownNodeParser(),
        persist_db_dir = PERSIST_DB_DIR
        ) -> VectorStoreIndex:
    
    FILENAME = "llama_index_blog_posts.csv"
    DATA_DIR = "./data/"

    if overwriting_db:
        if os.path.exists(persist_db_dir):
            os.system(f"rm -r {persist_db_dir}")
            print(f"Deleted local db on {persist_db_dir}")
        os.makedirs(persist_db_dir)

    # init the pipeline with transformations
    ingestion_pipeline = IngestionPipeline(
        transformations=[
            chunker,
            embedding_model,        
        ],
        vector_store=vector_store,
        docstore=docstore,
    )

    # load the documents
    documents = csv_processor.csv_load(DATA_DIR + FILENAME)
    print(f"Loaded {len(documents)} documents.")

    # run the pipeline to get nodes
    nodes = ingestion_pipeline.run(
        documents=documents,
        in_place=True,
        show_progress=True,
    )
    print(f"Created {len(nodes)} nodes.")

    # init storage context
    storage_context = StorageContext.from_defaults(
        docstore=docstore,
        vector_store=vector_store,
        index_store=index_store,
    )

    # create (or load) docstore and add nodes
    storage_context.docstore.add_documents(nodes)

    # build index + save index
    print("Building indexes and persist to local storage...")
    vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
    vector_index.set_index_id("vector_index")
    vector_index.storage_context.persist(persist_dir=persist_db_dir)

    print(f"Saved the indexes to {persist_db_dir}.")
    return vector_index

def load_index(
        docstore = SimpleDocumentStore(), 
        index_store=SimpleIndexStore(), 
        vector_store = SimpleVectorStore(), 
        persist_db_dir = PERSIST_DB_DIR
    ) -> VectorStoreIndex:

    storage_context = StorageContext.from_defaults(
        docstore=docstore.from_persist_dir(persist_db_dir),
        vector_store=vector_store.from_persist_dir(persist_db_dir, namespace="default"),
        index_store=index_store.from_persist_dir(persist_db_dir),
    )

    vector_index = load_index_from_storage(storage_context, index_id="vector_index")
    print(f"Loaded the index.")
    return vector_index