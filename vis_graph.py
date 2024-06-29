from llama_index.core import SimpleDirectoryReader,KnowledgeGraphIndex
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core import StorageContext
from pyvis.network import Network


documents = SimpleDirectoryReader(input_files = ['data/Solar_System_Exploration_and_India_contributions.pdf']).load_data()
print(len(documents))
print(f"Document Metadata: {documents[0].metadata}")

graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)

# NOTE: can take a while!
index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=2,
    storage_context=storage_context,
)
g = index.get_networkx_graph()
net = Network(notebook=True, cdn_resources="in_line", directed=True)
net.from_nx(g)
net.show("example.html")