

# News Chatbot Thesis Project
### (Detail here can be changed on the process of research and finding thoughtout the period of thesis)
## Overview

The **News Chatbot Thesis Project** aims to develop an intelligent conversational agent designed to provide users with up-to-date news, insightful analysis, and relevant information across various topics. Leveraging advanced Natural Language Processing (NLP) techniques and robust data management strategies, the chatbot ensures accurate and contextually relevant responses to user queries.

## Key Ideas and Approaches

1. **Database Duplication**
   - **Purpose:** Create a separate experimental MongoDB database to safely test and implement various chatbot functionalities without affecting the original dataset.
   - **Method:** Utilize Python with PyMongo to programmatically duplicate the existing MongoDB database.

2. **Keyword-Based Retrieval**
   - **Purpose:** Implement a traditional search mechanism to retrieve relevant news articles based on keyword matching.
   - **Method:** Use MongoDB's text indexes to perform keyword searches on news content.

3. **Semantic Search with FAISS**
   - **Purpose:** Enhance the chatbot's ability to understand and retrieve relevant information based on the semantic meaning of user queries.
   - **Method:** Implement FAISS (Facebook AI Similarity Search) to perform efficient similarity searches using embeddings generated by Sentence-BERT.

4. **Response Generation Models**
   - **Approach 1: Baseline Retrieval-Based Response**
     - Use retrieved articles as context to generate responses using generative models like GPT-4.
   - **Approach 2: Fine-Tuned Models**
     - Fine-tune pre-trained models such as BART or T5 to improve response accuracy and relevance.

5. **Retrieval-Augmented Generation (RAG)**
   - **Purpose:** Combine retrieval mechanisms with generative models to produce more informed and contextually rich responses.
   - **Method:** Utilize RAG models to integrate retrieved documents directly into the response generation process.

## Technologies Used

- **Programming Language:** Python
- **Backend Framework:** Flask
- **Databases:** MongoDB, FAISS
- **NLP Libraries:** Hugging Face Transformers, Sentence-Transformers, SpaCy
- **Others:** PyMongo, Pytest for testing


## Setup and Installation

### Prerequisites

- **Python:** Version 3.6 or higher
- **MongoDB:** Installed and running locally or accessible remotely
- **Git:** For version control and cloning the repository

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/news-chatbot.git
   cd news-chatbot
   ```

2. **Create a Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure MongoDB**
   - Ensure MongoDB is running locally or update the `mongo_uri` in the configuration files to point to your MongoDB instance.

5. **Initialize the Database Duplication and FAISS Index**
   ```bash
   python src/utils/db_utils.py
   python src/utils/faiss_utils.py
   ```

### Running the Application

Start the Flask server:
```bash
python src/app.py
```

Access the chatbot API at `http://localhost:5000/`.

## Usage

- **Duplicate Database:** Use the `/duplicate_db` endpoint to create an experimental copy of your primary database.
- **Semantic Search:** Use the `/search/semantic` endpoint to perform semantic searches based on user queries and retrieve relevant news articles.
- **Response Generation:** Utilize the response generation endpoints to receive contextually rich answers based on retrieved information.

## Future Work

- **User Interface:** Develop a user-friendly chat interface using Next.js for seamless interactions.
- **Knowledge Graph Integration:** Enhance the chatbot's understanding by representing relationships between entities using knowledge graphs.
- **Advanced Response Models:** Implement and fine-tune more sophisticated models to improve response quality and relevance.
- **Performance Optimization:** Continuously refine search algorithms and response generation processes for better efficiency and scalability.



## Contact

For any inquiries or feedback, please contact [roshandevkota1997@gmail.com](mailto:your.roshandevkota1997@gmail.com).

