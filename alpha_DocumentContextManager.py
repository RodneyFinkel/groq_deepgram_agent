# Using chromadb
from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer # For better embeddings
import numpy as np
import time
import logging

# New Imports for Hybrid Search
from rank_bm25 import BM25Okapi
from ragatouille import RAGPretrainedModel  # For ColBERT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentContextManager:
    def __init__(self, similarity_threshold=0.1):
        # Using chromadb
        self.client = Client(Settings(persist_directory="./chroma_storage", anonymized_telemetry=False))
        print("Chroma Initialized")
        self.collection = self.client.get_or_create_collection("documents", metadata={"hnsw:space": "cosine"}) # Ensure cosine similarity is used
        
        self.model = SentenceTransformer('all-MiniLM-L6-v2') # New model
        print("SentenceTransformer Initialized")
        
        # Store similarity threshold
        self.similarity_threshold = similarity_threshold
        logging.info(f"Initialized with similarity threshold: {self.similarity_threshold}")
        
        # CHANGE: Initialize last_raw_results to store raw retrieval data for debugging
        self.last_raw_results = []
        self.retrieval_config = {
            'hybrid_enabled': True,  # Toggle hybrid search
            'semantic_weight': 0.7,   # Weight for semantic score in fusion (0-1)
            'bm25_weight': 0.3,       # Weight for BM25 score in fusion (0-1)
            'bm25_k1': 1.2,           # BM25 term saturation
            'bm25_b': 0.75,           # BM25 length normalization
            'rerank_enabled': True,  # Toggle ColBERT reranking
            'rerank_k': 50,           # Initial retrieve this many for reranking, then take top_k
            'colbert_model': 'colbert-ir/colbertv2.0'  # Pretrained ColBERT model
        }    
        
        # New: Preload ColBERT if enabled (lazy load on first use)
        self.colbert_reranker = None
        
        # New: BM25 index (built on add document)
        self.bm25_index = None
        self.documents_for_bm25 = [] # list of tokenized docs for BM25
        
        

    def set_similarity_threshold(self, threshold):  # NEW: Set the similarity threshold for document retrieval
        if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
            raise ValueError("Similarity threshold must be a number between 0 and 1")
        self.similarity_threshold = float(threshold)
        logging.info(f"Updated similarity threshold to: {self.similarity_threshold}")
        
    # New: Setters and getters for retrieval config
    def set_retrieval_config(self, config):
        self.retrieval_config.update(config)
        logging.info(f"Updated retrieval config: {self.retrieval_config}")
        if self.retrieval_config['rerank_enabled'] and not self.colbert_reranker:
            self.colbert_reranker = RAGPretrainedModel.from_pretrained(self.retrieval_config['colbert_model'])
            logging.info(f"Loaded ColBERT model: {self.retrieval_config['colbert_model']}")
            
    def get_retrieval_config(self):
        return self.retrieval_config
        
    
    # NEW: using Sentence Transformer
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
        
        # New: Update BM25 index
        tokenized_doc = clean_text.lower().split() # simple tokenization for BM25
        self.documents_for_bm25.append(tokenized_doc)
        self.bm25_index = BM25Okapi(self.documents_for_bm25) # rebuild index (efficient for small corpora, optimize for large)
        logging.info(f"Updated BM25 index with new document {doc_id}")

    
    #  Using Chromadb
    def get_similar_documents(self, query, top_k=10, keyword_filter=None):
        if len(query.strip()) < 3: # skip very short queries
            print('~Query to short, skipping retrieval.')
            return []
        
        # query_embedding = self._embed_text(query).tolist()
        query_embedding = self._embed_text(query)
        query_params = {
            'query_embeddings': [query_embedding],
            'n_results': top_k,
            'include': ["documents", "metadatas", "distances"]
        }
        
        results = self.collection.query(**query_params)
        
        # Extract inner lists (Chroma returns nested lists for multi-query, but we have one query)
        ids = results["ids"][0] if results["ids"] else []
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else [] # NEW
        
        # NEW: store raw results for debugging and ui
        self.last_raw_results = [
            {
                "doc_id": ids[i],
                "distance": distances[i],
                "similarity": 1 - distances[i],
                "filename": metadatas[i].get("filename", "Unknown") if metadatas else "Unknown",
                "snippet": documents[i][:100] + "..." if documents and len(documents[i]) > 100 else documents[i]
            } for i in range(len(ids))
        ]
        
        # New: Hybrid Search if enabled
        hybrid_scores = {}
        if self.retrieval_config['hybrid_enabled'] and self.bm25_index:
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25_index.get_scores(tokenized_query)
            for i, doc_id in enumerate(ids):
                semantic_sim = 1 - distances[i]
                bm25_score = bm25_scores[i] if i < len(bm25_scores) else 0 # Align with retrieved docs
                fused_score = (
                    self.retrieval_config['semantic_weight'] * semantic_sim +
                    self.retrieval_config['bm25_weight'] * bm25_score
                )
                hybrid_scores[doc_id] = fused_score
                
                # sort by fused top score and take top k
                sorted_docs = sorted(hybrid_scores.items(), key=lambda x:[1], reverse=True)[:top_k]
                ids = [doc[0] for doc in sorted_docs]
                # Refetch docs, metas for sorted ids (inneficient, will optimize later)
                refetched = self.collection.get(ids=ids, include=['documents', 'metadatas'])
                documents = refetched['documents']
                metadatas = refetched['metadatas']
                
        # New: ColBERT reranking if enabled
        if self.retrieval_config['rerank_enabled'] and self.colbert_reranker:
            # Prepare docs for reranking
            rerank_docs = documents[:self.retrieval_config['rerank_k']]
            reranked = self.colbert_reranker.rerank(query, rerank_docs, k=top_k)
            # Update with reranked order/scores
            documents = [doc['content'] for doc in reranked]
            metadatas = [metadatas[rerank_docs.index(doc['content'])] for doc in reranked]
                            
                       
        
        # NEW: Similarity threshold filtering
        similar_docs = []
        for i in range(len(ids)):
            similarity = 1 - distances[i] if distances else 0 
            logging.info(f"Raw distance for {ids[i]}: {distances[i]}, similarity: {similarity}") # Log raw scores
            if similarity >= self.similarity_threshold:
                similar_docs.append({
                    "doc_id": ids[i],
                    "document": documents[i],
                    "metadata": metadatas[i],
                    "similarity": similarity # Include similirity score if available
                })
            else:
                logging.debug(f"Document {ids[i]} filtered out(similarity: {similarity} < {self.similarity_threshold})")
                
        
        logging.info(f"Retrieved {len(similar_docs)} documents with similarity >= {self.similarity_threshold}")
        return similar_docs


