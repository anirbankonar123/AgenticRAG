from llama_index.core import SimpleDirectoryReader
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.indices.property_graph import TextToCypherRetriever
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.mistralai import MistralAI
from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor


documents = SimpleDirectoryReader(input_files = ['data/Solar_System_Exploration_and_India_contributions.pdf']).load_data()

embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
llm = MistralAI(model="mistral-large-latest")
vec_store = SimpleVectorStore()

kg_extractor = SimpleLLMPathExtractor(llm=llm)

index = PropertyGraphIndex.from_documents(documents, kg_extractors=[kg_extractor],vector_store=vec_store)

sub_retriever = TextToCypherRetriever(index.property_graph_store, llm=llm)

query_engine = index.as_query_engine(
    sub_retrievers=[
       sub_retriever,

    ],
    llm=llm,
)

response = query_engine.query("What is India's achievement in the latest space mission ? Give details of the most Recent Project and related projects, in few paragraphs?")




