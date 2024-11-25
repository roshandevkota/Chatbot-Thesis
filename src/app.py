# finance-chatbot/src/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.db_utils import connect_to_mongo
import logging
import sys
import spacy
from collections import defaultdict
import re
from flask_caching import Cache

# Configure logging with RotatingFileHandler to prevent log files from growing indefinitely
from logging.handlers import RotatingFileHandler

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Configure Cache
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Configure Logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create handlers
console_handler = logging.StreamHandler(sys.stdout)
file_handler = RotatingFileHandler('app.log', maxBytes=1000000, backupCount=5)

# Create formatters and add them to handlers
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Define duplication parameters
SOURCE_DB = "news_db"        # Your main database name
TARGET_DB = "experiment_db"  # Experimental database name
MONGO_URI = "mongodb://localhost:27017/"  # MongoDB connection URI

# Initialize MongoDB connection
client = connect_to_mongo(MONGO_URI)
logger.info("Connected to MongoDB successfully.")

# Load spaCy large English model
try:
    nlp = spacy.load("en_core_web_lg")
    logger.info("spaCy 'en_core_web_lg' model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load spaCy model 'en_core_web_lg': {e}")
    sys.exit(1)

def extract_keywords(text, nlp, top_n=10):
    """
    Extracts keywords from the given text using spaCy's noun chunks.

    Parameters:
    - text (str): Text to extract keywords from.
    - nlp (spacy.lang): spaCy language model.
    - top_n (int): Number of top keywords to return.

    Returns:
    - list: List of extracted keywords.
    """
    doc = nlp(text.lower())
    keywords = [chunk.text for chunk in doc.noun_chunks]
    freq = defaultdict(int)
    for word in keywords:
        freq[word] += 1
    sorted_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in sorted_keywords[:top_n]]

def create_text_indexes(db_name, collection_name, fields, weights=None, default_language="english"):
    """
    Creates a text index on specified fields in a MongoDB collection.

    Parameters:
    - db_name (str): Name of the database.
    - collection_name (str): Name of the collection.
    - fields (list): List of fields to include in the text index.
    - weights (dict, optional): Dictionary specifying the weight of each field.
    - default_language (str): Language for the text index.
    """
    db = client[db_name]
    collection = db[collection_name]
    
    # Create a list of tuples for fields with "text" type
    index_fields = [(field, "text") for field in fields]
    
    index_options = {
        "name": "TextIndex",
        "default_language": default_language
    }
    
    if weights:
        index_options["weights"] = weights
    
    try:
        collection.create_index(index_fields, **index_options)
        logger.info(f"Text index created on '{collection_name}' in '{db_name}'.")
    except Exception as e:
        logger.error(f"Failed to create text index on '{collection_name}' in '{db_name}': {e}")

def duplicate_database_with_keywords(source_db, target_db, client, nlp):
    """
    Duplicates a MongoDB database from source_db to target_db, preserving existing keywords and adding missing ones.

    Parameters:
    - source_db (str): Name of the source database.
    - target_db (str): Name of the target database.
    - client (pymongo.MongoClient): MongoDB client instance.
    - nlp (spacy.lang): spaCy language model.
    """
    try:
        src = client[source_db]
        tgt = client[target_db]
        tgt.client.drop_database(target_db)
        logger.info(f"Dropped existing database '{target_db}'.")

        # Iterate through collections in the source database and copy data
        for collection_name in src.list_collection_names():
            src_collection = src[collection_name]
            tgt_collection = tgt[collection_name]
            data = list(src_collection.find())
            for doc in data:
                # Check if 'keywords' exist and are non-empty
                existing_keywords = doc.get('keywords', [])
                if not existing_keywords:
                    content = doc.get('content', '')
                    title = doc.get('title', '')
                    combined_text = f"{title} {content}"
                    keywords = extract_keywords(combined_text, nlp)
                    doc['keywords'] = keywords
                # If 'keywords' exist, retain them
            if data:
                tgt_collection.insert_many(data)
                logger.info(f"Copied {len(data)} documents to '{target_db}.{collection_name}'.")
    except Exception as e:
        logger.error(f"Failed to duplicate database with keywords: {e}")
        raise e

def summarize_content(content, nlp, max_sentences=2):
    """
    Summarizes the content by extracting the first few sentences.

    Parameters:
    - content (str): Full content of the article.
    - nlp (spacy.lang): spaCy language model.
    - max_sentences (int): Number of sentences to include in the summary.

    Returns:
    - str: Summarized content.
    """
    doc = nlp(content)
    sentences = list(doc.sents)
    summary = ' '.join([sent.text for sent in sentences[:max_sentences]])
    return summary

@app.route('/duplicate_db', methods=['POST'])
def duplicate_db():
    """
    API endpoint to trigger database duplication.
    Duplicates SOURCE_DB to TARGET_DB, preserves existing keywords, and creates text indexes.
    """
    logger.info(f"Received request to duplicate DB from '{SOURCE_DB}' to '{TARGET_DB}'")
    
    try:
        # Duplicate the database and preserve existing keywords
        duplicate_database_with_keywords(SOURCE_DB, TARGET_DB, client, nlp)
        logger.info("Database duplication with keyword preservation completed successfully.")
        
        # Create text indexes on the duplicated collection
        fields_to_index = ["title", "meta_description", "keywords", "authors", "content"]
        weights = {
            "title": 5,
            "meta_description": 3,
            "keywords": 4,
            "authors": 2,
            "content": 1
        }
        create_text_indexes(TARGET_DB, "scraped_articles", fields_to_index, weights=weights, default_language="english")
        
        logger.info(f"Text indexes created on '{TARGET_DB}.scraped_articles'.")
        
        return jsonify({"message": f"Database duplicated from '{SOURCE_DB}' to '{TARGET_DB}' successfully."}), 200
    except Exception as e:
        logger.error(f"Database duplication failed: {e}")
        return jsonify({"error": "Database duplication failed.", "details": str(e)}), 500

@app.route('/search/keyword', methods=['POST'])
# @cache.cached(timeout=300, query_string=True)  # Cache results for 5 minutes
def search_keyword():
    """
    API endpoint to perform keyword-based search using MongoDB's text search.
    Expects JSON payload with 'query' and 'top_k'.
    """
    data = request.get_json()
    query = data.get('query', '').strip()
    top_k = data.get('top_k', 5)
    print('top_k', top_k)
    
    # Validate 'top_k' parameter
    if not isinstance(top_k, int) or top_k <= 0:
        return jsonify({'error': 'Invalid top_k value. It must be a positive integer.'}), 400
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    logger.info(f"Received keyword search request for query: '{query}' with top_k: {top_k}")
    
    try:
        # Perform text search leveraging field_weights
        target_db = client[TARGET_DB]
        collection = target_db['scraped_articles']  # Updated collection name
        
        # Execute text search
        cursor = collection.find(
            {"$text": {"$search": query}},
            {"score": {"$meta": "textScore"}, "title": 1, "content": 1, "authors":1, "publish_date":1, "canonical_link":1, "keywords":1, "meta_description":1}
        ).sort([("score", {"$meta": "textScore"})]).limit(top_k)
        
        results = []
        for doc in cursor:
            # Optional: Summarize content
            summary = summarize_content(doc.get('content', ''), nlp)
            results.append({
                'id': str(doc['_id']),
                'title': doc.get('title'),
                'summary': summary,
                'authors': doc.get('authors'),
                'publish_date': doc.get('publish_date'),
                'canonical_link': doc.get('canonical_link'),
                'keywords': doc.get('keywords'),
                'meta_description': doc.get('meta_description'),
                'relevance_score': doc.get('score')  # textScore based on field_weights
            })
        
        if not results:
            return jsonify({'message': 'No relevant articles found.'}), 404
        
        return jsonify({'results': results}), 200
    except Exception as e:
        logger.error(f"Keyword search failed: {e}")
        return jsonify({'error': 'Keyword search failed.', 'details': str(e)}), 500

if __name__ == '__main__':
    try:
        # Specify the corrected port
        app.run(debug=True, port=5002)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Flask app terminated.")
