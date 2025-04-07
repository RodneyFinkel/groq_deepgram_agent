# Using chromadb
from chromadb import Client
from chromadb.config import Settings
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time


class DocumentContextManager:
    def __init__(self):
        # Using chromadb
        self.client = Client(Settings(persist_directory="./chroma_storage", anonymized_telemetry=False))
        print("Chroma Initialized")
        self.collection = self.client.get_or_create_collection("documents")
        
        # Load pre-trained model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        print("Bert initialized")
        
    def _embed_text(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Mean pooling to get a single vector for the document
        embeddings = outputs.last_hidden_state.mean(dim=1)
        if embeddings is None or embeddings.shape[0] == 0:
            raise ValueError("Emebeddings generation failed for text.")
        print(f"Generated Embedding Shape: {embeddings.shape}")
        return embeddings.cpu().numpy().flatten()
    
    # Using chromadb
    def add_document(self, doc_id, text, filename):
        clean_text = " ".join(text.split()) # clean up document text
        embedding = self._embed_text(clean_text)
        metadata = {
            "filename": filename,
            "upload_time": time.time(),
            "summary":text[:50]
        }
        
        print(f"Storing Embedding for Doc ID: {doc_id} with embedding:{embedding[:5]}")
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata],
            documents=[clean_text]
        )

    
    #  Using chromadb
    
    def get_similar_documents(self, query, top_k=1):
        query_embedding = self._embed_text(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "embeddings"]
        )
        
        # Ensure results contain valid data
        similar_docs = []
        for doc_id, document, metadata in zip(results["ids"], results["documents"], results["metadatas"]):
            if doc_id and document:  # Check if both doc_id and document exist
                similar_docs.append({
                    "doc_id": doc_id[0],  # Extract the first item
                    "document": document[0],  # Flatten the document list
                    "metadata": metadata  # Metadata is typically a single dictionary
                })
        return similar_docs


