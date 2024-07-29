# Using a DB
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from models import Document
# from app2 import db

class DocumentContextManager:
    def __init__(self):
        from models import Document
        from app2 import db
        # Load pre-trained model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def add_document(self, filename, text):
        embedding = self._embed_text(text)
        doc = Document(filename=filename, text=text)
        db.session.add(doc)
        db.session.commit()
        

    def _embed_text(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Mean pooling to get a single vector for the document
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()

    def get_similar_documents(self, query, top_k=5):
        query_embedding = self._embed_text(query)
        documents = Document.query.all()
        similarities = {}
        
        for doc in documents:
            doc_embedding = self._embed_text(doc.text)
            sim = cosine_similarity(query_embedding, doc_embedding)
            similarities[doc.id] = sim[0][0]
        
        sorted_docs = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
        return [(doc_id, Document.query.get(doc_id).text) for doc_id, _ in sorted_docs[:top_k]]


