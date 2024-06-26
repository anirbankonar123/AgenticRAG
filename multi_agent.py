from llama_index.core import SimpleDirectoryReader,VectorStoreIndex,SummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core import StorageContext
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings
from llama_index.llms.mistralai import MistralAI
from typing import List,Optional
from llama_index.core.vector_stores import MetadataFilters,FilterCondition
from llama_index.core.tools import FunctionTool,QueryEngineTool

# import  nest_asyncio
# nest_asyncio.apply()

documents = SimpleDirectoryReader(input_files = ['data/Solar_System_Exploration_and_India_contributions.pdf']).load_data()
print(len(documents))
print(f"Document Metadata: {documents[0].metadata}")

splitter = SentenceSplitter(chunk_size=1024,chunk_overlap=100)
nodes = splitter.get_nodes_from_documents(documents)
print(f"Length of nodes : {len(nodes)}")
#print(f"get the content for node 0 :{nodes[0].get_content(metadata_mode='all')}")


db = chromadb.PersistentClient(path="./chroma_db_mistral")
chroma_collection = db.get_or_create_collection("multidocument-agent")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

print("getting embedding model")
embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model
Settings.chunk_size = 1024

print("creating instance of LLM")
llm = MistralAI(model="mistral-large-latest")

print("instantiating Vector store")
name = "SolarSystem_India" #BERT_arxiv"
vector_index = VectorStoreIndex(nodes,storage_context=storage_context)
vector_index.storage_context.vector_store.persist(persist_path="chroma_db")
#
# Define Vectorstore Autoretrieval tool
def vector_query(query:str,page_numbers:Optional[List[str]]=None)->str:
    '''
    perform vector search over index on
    query(str): query string needs to be embedded
    page_numbers(List[str]): list of page numbers to be retrieved,
                          leave blank if we want to perform a vector search over all pages
    '''
    page_numbers = page_numbers or []
    metadata_dict = [{"key":'page_label',"value":p} for p in page_numbers]
    #
    query_engine = vector_index.as_query_engine(similarity_top_k =2,
                                              filters = MetadataFilters.from_dicts(metadata_dict,
                                                                                    condition=FilterCondition.OR)
                                              )
    #
    response = query_engine.query(query)
    return response
    #
print("create llamiondex FunctionTool ")
vector_query_tool = FunctionTool.from_defaults(name=f"vector_tool_{name}",
                                              fn=vector_query)
print("instantiating KG Index with summary")
summary_index = SummaryIndex(nodes)
summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize",
                                                      se_async=True,)

print("create summary tool")
summary_query_tool = QueryEngineTool.from_defaults(name=f"summary_tool_{name}",
                                                    query_engine=summary_query_engine,
                                                  description=("Use ONLY IF you want to get a holistic summary of the documents."
                                              "DO NOT USE if you have specified questions over the documents."))

print("predict using vector query tool")
response = llm.predict_and_call([summary_query_tool],
                                "Summarize the content in page number 10 to 15",
                                verbose=True)

from pyvis.network import Network



