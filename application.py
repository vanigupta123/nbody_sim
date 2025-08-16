import numpy as np
import matplotlib.pyplot as plt
import faiss
from langchain.document_loaders import UnstructuredPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

# load documents - primarily PDFs and HTML articles - some APIs
pdf_path = ""
url = ""
pdf_docs = UnstructuredPDFLoader(pdf_path).load()
web_docs = WebBaseLoader(url).load()
# preprocess, clean, extract metadata
    # standardize formatting
    # strip headers and footers, remove citations, etc.
    # important metadata: section headers for graph linking, custom tags (simulation type, domain, etc.)
# embed chunks, potentially using metadata
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pdf_docs)
embedder = OpenAIEmbeddings()
vectors_pdf = embedder.embed_documents([chunk.page_content for chunk in chunks])

if type(web_docs) == list():
    chunks = splitter.split_documents(web_docs)
    vectors_web = embedder.embed_documents([doc.page_content for doc in chunks])
# extract graph nodes
# store vectors or nodes in index

d = 64 # dimension
nb = 100000 # db size
nq = 10000 # number of queries
np.random.seed(1234) # causes random number generator to start at the same initial (seed) state
xb = np.random.random((nb, d)).astype('float32') # matrix that's 100,000x64
xb[:, 0] += np.arange(nb) / 1000. # xb[:, 0] has dimension 100,000x1
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

index = faiss.IndexFlatL2(d)
print(index.is_trained)
index.add(xb) # add vectors to index -- IDs are assigned to each vector
# other index types can store custom IDs, instead of assigning numerical order IDs
print(index.ntotal)

k = 4
D, I = index.search(xq, k) # integer matrix of size nq x k
print(D) # nq x k matrix, where row i contains the IDs of query vector i, sorted by increasing distance
print(I) # nq x k matrix of squared distances
print(I[:5])
print(I[-5:])

# index is partitioned into cells with centroids
    # IndexFlatL2 will search between the query vector and other vectors belonging in that specific cell
    # helps to reduce the scope of the search -- produces an approximate result, not exact
nlist = 50 # number of cells
quantizer = faiss.IndexFlatL2(d) 
index = faiss.IndexIVFFlat(quantizer, d, nlist)