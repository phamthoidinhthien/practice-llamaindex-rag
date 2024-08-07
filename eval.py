import pandas as pd
from tqdm import tqdm
from datasets import Dataset 
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas import evaluate
from llama_index.core.base.base_query_engine import BaseQueryEngine
from utils.retry import retry_with_exponential_backoff


def run_evaluation(query_engine:BaseQueryEngine, with_retry_support=False):
    
    print("... START EVALUATION PROCESS ....")
    print("Loading test set...")
    testset = pd.read_csv("./data/testset.csv")
    
    print("Loading index and query engine...")
    vector_query_engine = query_engine

    @retry_with_exponential_backoff
    def ask_question(**kwargs):
        return vector_query_engine.query(**kwargs)

    # Collecting answers
    answers = []
    source_answer_nodes = []
    
    input("Press Enter to start generating answers...")
    for question in tqdm(testset.question, desc="Generating answers..."):
        
        if with_retry_support:
            response = vector_query_engine.query(question)
        else:
            response = ask_question(str_or_query_bundle=question)
        
        answers.append(response.response)
        source_answer_nodes.append(response.source_nodes)

    testset["llm_answer"] = answers
    testset["source_answer_nodes"] = source_answer_nodes
    testset['contexts'] = testset['contexts'].apply(lambda x: eval(x) if isinstance(x, str) else [])
    testset['source_answer_nodes'] = testset['source_answer_nodes'].apply(lambda x: eval(x) if isinstance(x, str) else [])

    data_dict = {
        'question': testset.question.to_list(),
        'answer': testset.llm_answer.to_list(),
        'contexts' : testset.contexts.to_list(),
        'ground_truth': testset.ground_truth.to_list(),
        "source_answer_nodes": testset.source_answer_nodes.to_list(),
    }
    dataset = Dataset.from_dict(data_dict)

    input("Press Enter to start evaluating answers...")
    result = evaluate(
        dataset,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall,
        ])

    print("Evaluation result :", result)