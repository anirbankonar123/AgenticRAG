from llama_index.core import SimpleDirectoryReader,VectorStoreIndex,SummaryIndex,Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.llms.openai import OpenAI

embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model
Settings.chunk_size = 1024

print("creating instance of LLM")
Settings.llm = OpenAI(model="gpt-3.5-turbo")


documents = SimpleDirectoryReader(input_files = ['data/Solar_System_Exploration_and_India_contributions.pdf']).load_data()
print(len(documents))
print(f"Document Metadata: {documents[0].metadata}")

splitter = SentenceSplitter(chunk_size=1024,chunk_overlap=100)
nodes = splitter.get_nodes_from_documents(documents)
print(f"Length of nodes : {len(nodes)}")

summary_index = SummaryIndex(nodes)
vector_index = VectorStoreIndex(nodes)

summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
vector_query_engine = vector_index.as_query_engine()


summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description=(
        "Useful for summarization questions related to the document"
    ),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "Useful for retrieving specific context from the document."
    ),
)

query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        summary_tool,
        vector_tool,
    ],
    verbose=True
)

response = query_engine.query("What is India's achievement in the latest space mission ? Give details of the most Recent Project and related projects, in few paragraphs?")
print(str(response))
print(len(response.source_nodes))

response = query_engine.query("What is the unique achievement of Chandrayaan-3?")
print(str(response))
print(len(response.source_nodes))