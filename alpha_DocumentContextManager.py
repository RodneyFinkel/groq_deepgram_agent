# Using chromadb
from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer # For better embeddings
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentContextManager:
    def __init__(self):
        # Using chromadb
        self.client = Client(Settings(persist_directory="./chroma_storage", anonymized_telemetry=False))
        print("Chroma Initialized")
        self.collection = self.client.get_or_create_collection("documents")
        
        # Load pre-trained model and tokenizer
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.model = BertModel.from_pretrained('bert-base-uncased')
        # print("Bert initialized")
        self.model = SentenceTransformer('all-MiniLM-L6-v2') # New model
        print("SentenceTransformer Initialized")
        
        
    # def _embed_text(self, text):
    #     # Pre-tokenize without special tokens to control length precisely
    #     tokens = self.tokenizer.encode(text, add_special_tokens=False)
    #     if len(tokens) > 510:  # Leave room for [CLS] + [SEP] (~2 tokens)
    #         print(f"Warning: Input text truncated from {len(tokens)} to 510 tokens for BERT limit.")
    #         tokens = tokens[:510]
        
    #     # Re-encode with special tokens and padding
    #     inputs = self.tokenizer.decode(tokens, skip_special_tokens=True)  # Back to text, clean (NEW)
    #     inputs = self.tokenizer(inputs, return_tensors='pt', truncation=True, padding=True, max_length=512)
        
    #     print(f"Input token count (with special): {inputs['input_ids'].shape[1]}")  # Debug: Should be <=512
    #     with torch.no_grad():
    #         outputs = self.model(**inputs)
    #     # Mean pooling to get a single vector for the document
    #     embeddings = outputs.last_hidden_state.mean(dim=1)
    #     if embeddings is None or embeddings.shape[0] == 0:
    #         raise ValueError("Emebeddings generation failed for text.")
    #     print(f"Generated Embedding Shape: {embeddings.shape}")
    #     return embeddings.cpu().numpy().flatten()
    
    # NEW; using sentence transformer
    def _embed_text(self, text):
        embedding = self.model.encode(text, show_progress_bar=True)
        if isinstance(embedding, np.ndarray):
            #embedding = embedding.tolist()
            embedding = embedding
        elif not isinstance(embedding, list):
            logging.error(f"Unexpected embedding type: {type(embedding)}")
            raise ValueError(f"Unexpected embedding type: {type(embedding)}")
        logging.info(f"Generated Embedding Shape: {len(embedding)}")
        return embedding
    
    # Using chromadb
    def add_document(self, doc_id, text, filename):
        # Check for existing document to avoid duplicates
        try:
            existing = self.collection.get(ids=[doc_id])
            if existing['ids']:
                print(f"Doc {doc_id} already exists. Skipping addition.")
                return
        except:
            pass # Not found, proceed to add
        
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
    
    def get_similar_documents(self, query, top_k=10):
        if len(query.strip()) < 3: # skip very short queries
            print('~Query to short, skipping retrieval.')
            return []
        
        # query_embedding = self._embed_text(query).tolist()
        query_embedding = self._embed_text(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "embeddings"]
        )
        
        # Extract inner lists (Chroma returns nested lists for multi-query, but we have one query)
        ids = results["ids"][0] if results["ids"] else []
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        
        similar_docs = []
        for i in range(len(ids)):
            similar_docs.append({
                "doc_id": ids[i],
                "document": documents[i],
                "metadata": metadatas[i]
            })
        
        return similar_docs


