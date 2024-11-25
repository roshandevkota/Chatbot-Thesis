# finance-chatbot/src/utils/faiss_utils.py

import faiss
import numpy as np
from pymongo import MongoClient
from utils.db_utils import connect_to_mongo
import logging
import sys
from sentence_transformers import SentenceTransformer
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("faiss_utils.log")
    ]
)

class FAISSHandler:
    def __init__(self, db_name="experiment_db", collection_name="news_copy", index_file="faiss.index", mapping_file="id_mapping.pkl", embedding_model="all-MiniLM-L6-v2", mongo_uri="mongodb://localhost:27017/"):
        """
        Initializes the FAISS handler with the specified parameters.

        Parameters:
        - db_name (str): Name of the MongoDB database.
        - collection_name (str): Name of the collection containing news articles.
        - index_file (str): Path to save/load the FAISS index.
        - mapping_file (str): Path to save/load the ID mappings.
        - embedding_model (str): Pre-trained SentenceTransformer model name.
        - mongo_uri (str): MongoDB connection URI.
        """
        self.db_name = db_name
        self.collection_name = collection_name
        self.index_file = index_file
        self.mapping_file = mapping_file
        self.embedding_model = embedding_model
        self.mongo_uri = mongo_uri

        # Initialize MongoDB connection
        self.client = connect_to_mongo(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]

        # Initialize embedding model
        self.model = SentenceTransformer(self.embedding_model)
        logging.info(f"Loaded SentenceTransformer model: {self.embedding_model}")

        # Initialize FAISS index and ID mapping
        self.index = None
        self.id_mapping = []
        self.load_or_build_index()

    def load_or_build_index(self):
        """
        Loads the FAISS index and ID mapping from disk if available.
        Otherwise, builds the index from scratch.
        """
        try:
            # Load FAISS index
            self.index = faiss.read_index(self.index_file)
            logging.info(f"Loaded FAISS index from {self.index_file}")

            # Load ID mapping
            with open(self.mapping_file, 'rb') as f:
                self.id_mapping = pickle.load(f)
            logging.info(f"Loaded ID mapping from {self.mapping_file}")

        except (FileNotFoundError, faiss.FaissException) as e:
            logging.warning(f"FAISS index or ID mapping not found. Building a new index. Error: {e}")
            self.build_index()
            self.save_index()

    def build_index(self):
        """
        Builds the FAISS index from the existing data in the MongoDB collection.
        """
        # Retrieve all documents
        documents = self.collection.find({}, {'_id': 1, 'scraped_content': 1})
        texts = []
        ids = []

        for doc in documents:
            content = doc.get('scraped_content') or ""
            if content:
                texts.append(content)
                ids.append(str(doc['_id']))

        if not texts:
            logging.error("No documents with 'scraped_content' found in the collection.")
            sys.exit(1)

        # Generate embeddings
        logging.info("Generating embeddings for all documents...")
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        embeddings = embeddings.astype('float32')  # FAISS requires float32

        # Initialize FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)  # Simple flat (brute-force) index
        self.index.add(embeddings)
        logging.info(f"FAISS index built with {self.index.ntotal} vectors.")

        # Create ID mapping
        self.id_mapping = ids
        logging.info("ID mapping created.")

    def save_index(self):
        """
        Saves the FAISS index and ID mapping to disk.
        """
        faiss.write_index(self.index, self.index_file)
        logging.info(f"FAISS index saved to {self.index_file}")

        with open(self.mapping_file, 'wb') as f:
            pickle.dump(self.id_mapping, f)
        logging.info(f"ID mapping saved to {self.mapping_file}")

    def add_new_documents(self, documents):
        """
        Adds new documents to the FAISS index and updates the ID mapping.

        Parameters:
        - documents (list of dict): List of new documents to add.
        """
        texts = []
        ids = []

        for doc in documents:
            content = doc.get('scraped_content') or ""
            if content:
                texts.append(content)
                ids.append(str(doc['_id']))

        if not texts:
            logging.warning("No valid 'scraped_content' found in the new documents.")
            return

        # Generate embeddings
        logging.info(f"Generating embeddings for {len(texts)} new documents...")
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        embeddings = embeddings.astype('float32')

        # Add to FAISS index
        self.index.add(embeddings)
        logging.info(f"Added {len(embeddings)} vectors to the FAISS index.")

        # Update ID mapping
        self.id_mapping.extend(ids)
        logging.info("ID mapping updated with new document IDs.")

        # Save the updated index and mapping
        self.save_index()

    def search(self, query, top_k=5):
        """
        Searches the FAISS index for the most similar documents to the query.

        Parameters:
        - query (str): User query text.
        - top_k (int): Number of top similar documents to retrieve.

        Returns:
        - results (list of dict): List of retrieved documents with similarity scores.
        """
        # Generate embedding for the query
        logging.info("Generating embedding for the query...")
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype('float32')

        # Perform search
        distances, indices = self.index.search(query_embedding, top_k)
        logging.info(f"Search completed. Top {top_k} results retrieved.")

        # Retrieve documents from MongoDB
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.id_mapping):
                doc_id = self.id_mapping[idx]
                doc = self.collection.find_one({'_id': doc_id}, {'headline': 1, 'scraped_content': 1, 'publisher': 1, 'publish_time': 1, 'url': 1})
                if doc:
                    results.append({
                        'id': str(doc['_id']),
                        'headline': doc.get('headline'),
                        'content': doc.get('scraped_content'),
                        'publisher': doc.get('publisher'),
                        'publish_time': doc.get('publish_time'),
                        'url': doc.get('url'),
                        'similarity_score': float(distance)
                    })
        return results
