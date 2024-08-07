from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core import get_response_synthesizer,VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.base.base_query_engine import BaseQueryEngine
import datetime

vector_store_info = VectorStoreInfo(
    content_info="Receipts",
    metadata_info=[
        MetadataInfo(
            name="date",
            description="The result for this date",
            type=f"date in MMM dd, yyyy format. Today is {datetime.datetime.today()}",
        ),
        ]
    )

def get_query_engine(
    index: VectorStoreIndex,
    vector_store_info: VectorStoreInfo = vector_store_info,
    similarity_top_k: int = 2,
) -> BaseQueryEngine:
    
    retriever = VectorIndexAutoRetriever(
        index=index,
        vector_store_info=vector_store_info,
        similarity_top_k=similarity_top_k,
        empty_query_top_k=10,  # if only metadata filters are specified, this is the limit
        verbose=False,
    )

    response_synthesizer = get_response_synthesizer()

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer
    )

    return query_engine