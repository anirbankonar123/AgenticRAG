from llama_index.core import SimpleDirectoryReader,SummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core import StorageContext
from llama_index.llms.mistralai import MistralAI
from typing import List,Optional
from llama_index.core.vector_stores import MetadataFilters,FilterCondition
from llama_index.core.tools import FunctionTool, QueryEngineTool
import os
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
#

db = chromadb.PersistentClient(path="./chroma_db_solary")
chroma_collection = db.get_or_create_collection("multidocument-agent")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
def get_doc_tools(file_path:str,name:str)->str:
  '''
  get vector query and summary query tools from a document
  '''
  #load documents
  documents = SimpleDirectoryReader(input_files = [file_path]).load_data()
  splitter = SentenceSplitter(chunk_size=1024,chunk_overlap=100)
  nodes = splitter.get_nodes_from_documents(documents)
  print(f"Length of nodes : {len(nodes)}")
  #instantiate Vectorstore
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
  #llamiondex FunctionTool wraps any python function we feed it
  vector_query_tool = FunctionTool.from_defaults(name=f"vector_tool_{name}",
                                                fn=vector_query)
  # Prepare Summary Tool
  summary_index = SummaryIndex(nodes)
  summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize",
                                                       se_async=True,)
  summary_query_tool = QueryEngineTool.from_defaults(name=f"summary_tool_{name}",
                                                     query_engine=summary_query_engine,
                                                    description=("Use ONLY IF you want to get a holistic summary of the documents."
                                                "DO NOT USE if you have specified questions over the documents."))
  return vector_query_tool,summary_query_tool


root_path = "data"
file_name = []
file_path = []
for file in os.listdir(root_path):
  if file.endswith(".pdf"):
    file_name.append(file.split(".")[0])
    file_path.append(os.path.join(root_path,file))
#
print(file_name)
print(file_path)

papers_to_tools_dict = {}
for name,filename in zip(file_name,file_path):
  vector_query_tool,summary_query_tool = get_doc_tools(filename,name)
  papers_to_tools_dict[name] = [vector_query_tool,summary_query_tool]

initial_tools = [t for f in file_name for t in papers_to_tools_dict[f]]
print(initial_tools)

obj_index = ObjectIndex.from_objects(initial_tools,index_cls=VectorStoreIndex)
obj_retriever = obj_index.as_retriever(similarity_top_k=2)

# tools = obj_retriever.retrieve("What is India's achievement in the latest space mission ? Give details of the Project, in few paragraphs")
# # #
# print(tools[0].metadata)
# print(tools[1].metadata)


llm = MistralAI(model="mistral-large-latest")
agent_worker = FunctionCallingAgentWorker.from_tools(tool_retriever=obj_retriever,
                                                     llm=llm,
                                                     system_prompt="""You are an agent designed to answer queries over a scientific document.
                                                     Please always use the tools provided to answer a question.Do not rely on prior knowledge.""",
                                                     verbose=True)
agent = AgentRunner(agent_worker)

# response = agent.query("Compare and contrast self rag and crag.")
# print(str(response))

# response = agent.query("Summarize the paper corrective RAG.")

#response = agent.query("What is India's contribution to Solar system exploration")

response = agent.query("What is India's achievement in the latest space mission ? Give details of the most Recent Project and related projects, in few paragraphs")

print(response)
