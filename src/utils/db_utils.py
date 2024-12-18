# finance-chatbot/src/utils/db_utils.py

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("db_utils.log")
    ]
)

def connect_to_mongo(uri="mongodb://localhost:27017/"):
    """
    Establishes a connection to the MongoDB server.

    Parameters:
    - uri (str): MongoDB connection URI.

    Returns:
    - client (MongoClient): MongoDB client instance.
    """
    try:
        client = MongoClient(uri)
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ismaster')
        logging.info("Connected to MongoDB successfully.")
        return client
    except ConnectionFailure as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        sys.exit(1)

def empty_database(target_db):
    """
    Empties the specified MongoDB database by dropping all its collections.

    Parameters:
    - target_db (Database): The MongoDB database instance to empty.
    """
    try:
        collection_names = target_db.list_collection_names()
        for collection_name in collection_names:
            target_db.drop_collection(collection_name)
            logging.info(f"Dropped collection: {collection_name}")
        logging.info(f"Database '{target_db.name}' has been emptied.")
    except Exception as e:
        logging.error(f"Error while emptying the database '{target_db.name}': {e}")
        sys.exit(1)

def duplicate_database(source_db_name, target_db_name, client):
    """
    Duplicates a MongoDB database by copying all collections and their indexes.

    Parameters:
    - source_db_name (str): Name of the source database to copy from.
    - target_db_name (str): Name of the target database to copy to.
    - client (MongoClient): Connected MongoDB client instance.
    """
    source_db = client[source_db_name]
    target_db = client[target_db_name]

    # Empty the target database before duplication
    logging.info(f"Emptying the target database: {target_db_name}")
    empty_database(target_db)

    # Retrieve all collection names from the source database
    collection_names = source_db.list_collection_names()
    logging.info(f"Collections to copy: {collection_names}")

    for collection_name in collection_names:
        logging.info(f"\nCopying collection: {collection_name}")
        source_collection = source_db[collection_name]
        target_collection = target_db[collection_name]

        # Copy all documents from source to target collection in batches
        documents = source_collection.find()
        count = 0
        batch = []
        batch_size = 1000  # Adjust based on memory and performance
        for doc in documents:
            batch.append(doc)
            if len(batch) == batch_size:
                try:
                    target_collection.insert_many(batch)
                    count += len(batch)
                    logging.info(f"Inserted {count} documents into '{collection_name}'")
                except Exception as e:
                    logging.error(f"Error inserting documents into '{collection_name}': {e}")
                batch = []
        if batch:
            try:
                target_collection.insert_many(batch)
                count += len(batch)
                logging.info(f"Inserted {count} documents into '{collection_name}'")
            except Exception as e:
                logging.error(f"Error inserting documents into '{collection_name}': {e}")

        # Copy indexes from source to target collection
        logging.info(f"Copying indexes for collection: {collection_name}")
        indexes = source_collection.list_indexes()
        for index in indexes:
            index_dict = index.to_dict()
            # Skip the default index on _id as MongoDB creates it automatically
            if index_dict.get('name') == '_id_':
                continue
            # Remove any unique identifiers that should not be duplicated
            index_keys = index_dict.get('key')
            index_options = {k: v for k, v in index_dict.items() if k not in ['key', 'v', 'ns', 'name']}
            try:
                target_collection.create_index(list(index_keys.items()), **index_options)
                logging.info(f"Created index: {index_dict.get('name')}")
            except Exception as e:
                logging.error(f"Failed to create index '{index_dict.get('name')}': {e}")

    logging.info(f"\nDatabase duplication from '{source_db_name}' to '{target_db_name}' completed successfully.")
