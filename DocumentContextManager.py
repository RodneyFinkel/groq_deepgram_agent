import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class DocumentContextManager:
    def __init__(self):
        self.documents = {}
        self.embeddings = {}
        
        # Load pre-trained model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def add_document(self, doc_id, text):
        self.documents[doc_id] = text
        embedding = self._embed_text(text)
        self.embeddings[doc_id] = embedding

    def _embed_text(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Mean pooling to get a single vector for the document
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()

    def get_similar_documents(self, query, top_k=5):
        query_embedding = self._embed_text(query)
        similarities = {}
        
        for doc_id, doc_embedding in self.embeddings.items():
            sim = cosine_similarity(query_embedding, doc_embedding)
            similarities[doc_id] = sim[0][0]
        
        sorted_docs = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
        return sorted_docs[:top_k]
