# Using chromadb
from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer # Upgrade from BERT embeddings
import numpy as np
import time
import logging
import uuid

# New Imports for Hybrid Search
from rank_bm25 import BM25Okapi
from ragatouille import RAGPretrainedModel  # For ColBERT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentContextManager:
    def __init__(self, similarity_threshold=0.15):
        self.id = str(uuid.uuid4())
        self.client = Client(Settings(persist_directory="./chroma_storage", anonymized_telemetry=False))
        logging.info("Chroma Initialized")
        self.collection = self.client.get_or_create_collection("documents", metadata={"hnsw:space": "cosine"}) # Ensure cosine similarity is used
        
        self.model = SentenceTransformer('all-MiniLM-L6-v2') # New model
        logging.info("SentenceTransformer Initialized")
        
        # Store similarity threshold
        self.similarity_threshold = similarity_threshold
        logging.info(f"Initialized with similarity threshold: {self.similarity_threshold}")
        
        # CHANGE: Initialize last_raw_results to store raw retrieval data for debugging
        self.last_raw_results = []
        logging.info("initializing self.last_raw_results-pre get similar docs")
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
        
        logging.info(f"Storing Embedding for Doc ID: {doc_id} with embedding:{embedding[:5]}")
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata],
            documents=[clean_text]
        )
        
        # New: Update BM25 index (THIS IS WHERE THE bm25_index is created using the derired context_manager instance)
        tokenized_doc = clean_text.lower().split() # simple tokenization for BM25
        self.documents_for_bm25.append(tokenized_doc)
        logging.info(f"Tokenized document {doc_id}: {tokenized_doc[:10]}...(total {len(tokenized_doc)} tokens)")
        try:
            self.bm25_index = BM25Okapi(self.documents_for_bm25) # rebuild index (efficient for small corpora, optimize for large)
            logging.info(f"Successfully updated BM25 index with {len(self.documents_for_bm25)} documents")
            print(self.bm25_index)
        except Exception as e:
            logging.error(f"Failed to update BM25 index: {str(e)}")
            self.bm25_index = None
            
        
    def normalize_bm25_scores(self, bm25_scores):
        if len(bm25_scores) == 0:
            logging.info("No BM25 scores to normalize (empty list)")
            return bm25_scores
        min_score = min(bm25_scores)
        max_score = max(bm25_scores)
        if max_score == min_score:
            logging.info(f"BM25 scores identical: min={min_score}, max={max_score}, returning zeros")
            return [0.0] * len(bm25_scores)
        normalized_scores = [(score - min_score)/ (max_score - min_score) for score in bm25_scores]
        logging.info(f"BM25 scores normalized: min={min_score}, max={max_score}")
        return normalized_scores
    
    # Add this method to DocumentContextManager class in alpha_DocumentContextManager.py
    def rebuild_bm25_from_chroma(self):
        all_data = self.collection.get(include=['documents'])
        self.documents_for_bm25 = []
        for doc in all_data['documents']:
            tokenized_doc = doc.lower().split()
            if tokenized_doc:
                self.documents_for_bm25.append(tokenized_doc)
        if self.documents_for_bm25:
            try:
                self.bm25_index = BM25Okapi(self.documents_for_bm25)
                logging.info(f"Rebuilt BM25 index from {len(self.documents_for_bm25)} existing documents")
            except Exception as e:
                logging.error(f"Failed to rebuild BM25 from Chroma: {str(e)}")
    

    
    #  Using Chromadb
    def get_similar_documents(self, query, top_k=10, keyword_filter=None):
        logging.info("initiating get_similar_documents function")
        print(self.bm25_index)
        if len(query.strip()) < 3: # skip very short queries
            print('~Query to short, skipping retrieval.')
            return []
        
        # query_embedding = self._embed_text(query).tolist()
        query_embedding = self._embed_text(query)
        query_params = {
            'query_embeddings': [query_embedding],
            'n_results': top_k if not self.retrieval_config['rerank_enabled'] else self.retrieval_config['rerank_k'],
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
                "snippet": documents[i][:100] + "..." if documents and len(documents[i]) > 100 else documents[i],
                "bm25_score": 0.0 # Placeholder: updated below
            } for i in range(len(ids))
        ]
        logging.info("initializing self.last_raw_results in get_similar_documents")
        
        # Search to see if singleton in LanguageModelProcessor uses correct context_manager instance/object
        print(self.retrieval_config)
        print(self.bm25_index)
        logging.info(f"Documents for BM25: {len(self.documents_for_bm25)}")
        logging.debug(f"BM25 index type: {type(self.bm25_index)}")
        # New Hybrid Search
        if self.retrieval_config['hybrid_enabled'] and self.bm25_index:
            logging.info("Starting hybrid retrieval")
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25_index.get_scores(tokenized_query)
            logging.info(f"Raw BM25 scores: {bm25_scores[:5]}... (total {len(bm25_scores)})")
            normalized_bm25_scores = self.normalize_bm25_scores(bm25_scores)
            hybrid_scores = {}
            for i, doc_id in enumerate(ids):
                semantic_sim = 1 - distances[i]
                bm25_score = normalized_bm25_scores[i] if i < len(normalized_bm25_scores) else 0 # Align with retrieved docs
                self.last_raw_results[i]["bm25_score"] = bm25_score # Store normalized BM25 score
                fused_score = (
                    self.retrieval_config['semantic_weight'] * semantic_sim +
                    self.retrieval_config['bm25_weight'] * bm25_score
                )
                hybrid_scores[doc_id] = fused_score
                logging.info(f"Doc {doc_id}: semantic={semantic_sim:.4f}, bm25={bm25_score:.4f}, fused={fused_score:.4f}")
                
            # sort by fused top score and take top k # FIXED: dropped refetch docs everytime
            sorted_docs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            # Reorder results based on sorted doc_ids
            sorted_ids = [doc[0] for doc in sorted_docs]
            sorted_documents = []
            sorted_metadatas = []
            sorted_distances = []
            for doc_id in sorted_ids:
                idx = ids.index(doc_id)
                sorted_documents.append(documents[idx])
                sorted_metadatas.append(metadatas[idx])
                sorted_distances.append(1 - hybrid_scores[doc_id])  # Convert fused score back to distance for consistency
            ids = sorted_ids
            documents = sorted_documents
            metadatas = sorted_metadatas
            # Update distances to reflect fused scores for consistency
            distances = sorted_distances
        else:
            logging.info("Hybrid search skipped: hybrid_enabled=%s, bm25_index=%s",
                         self.retrieval_config['hybrid_enabled'], self.bm25_index is not None)        
            
                
        # New: ColBERT reranking if enabled
        if self.retrieval_config['rerank_enabled'] and self.colbert_reranker:
            # Prepare docs for reranking
            rerank_docs = documents[:self.retrieval_config['rerank_k']]
            reranked = self.colbert_reranker.rerank(query, rerank_docs, k=top_k)
            # Update with reranked order/scores
            documents = [doc['content'] for doc in reranked]
            metadatas = [metadatas[rerank_docs.index(doc['content'])] for doc in reranked]
            # Update distances to reflect ColBERT scores
            distances = [1 - doc['score'] for doc in reranked]
            ids = [ids[rerank_docs.index(doc['content'])] for doc in reranked]                
                       
        
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


