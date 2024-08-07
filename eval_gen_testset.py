from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai.utils import ALL_AVAILABLE_MODELS, CHAT_MODELS
from dotenv import load_dotenv
import os
import utils.csv_processor as csv_processor

load_dotenv('.env')
OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]
DATA_DIR = "./data/"
FILENAME = "llama_index_blog_posts.csv"
ALL_AVAILABLE_MODELS["gpt-4o-mini"]= 128000
CHAT_MODELS["gpt-4o-mini"] = 128000

documents = csv_processor.csv_load(DATA_DIR + FILENAME)
print(f"Loaded {len(documents)} documents.")

# generator with openai models
generator_llm = OpenAI(model="gpt-4o-mini")
critic_llm = OpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbedding()

generator = TestsetGenerator.from_llama_index(
    generator_llm,
    critic_llm,
    embeddings
)

distributions = {
    simple: 0.5,
    multi_context: 0.4,
    reasoning: 0.1
}

testset = generator.generate_with_llamaindex_docs(
    documents=documents, 
    test_size=100, 
    distributions=distributions
)

testset = testset.to_pandas()
testset.to_csv("./data/testset.csv")