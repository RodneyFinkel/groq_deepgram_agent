from elasticsearch import elasticsearch
from sentence_transformer import SentenceTransformer
import fitz

es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

model = SentenceTransformer('all-MiniLM-L6-v2')

def index_text_chunks(text_chunks, index_name='pdf_text'):
    for i, chunk in enumerate(text_chunks):
        embedding = model.encode(chunk).to_list()
        doc = {
            'text': chunk,
            'embedding': embedding
        }
        es.index(index=index_name, id=1, document=doc)