import os
import json
import requests
from dotenv import load_dotenv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from code_files.jiragit.cluster_knowledge import cluster_task_data
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from code_files.jiragit.cluster_employee import process_all_employee_data
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb

load_dotenv()

class EmployeeDBBot:
    def __init__(self):
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('PILA_GROQ_KEY')}"
        }
        self.model = "llama-3.3-70b-versatile"
        self.temperature = 0
        self.max_tokens = 1024
        self.total_token_usage = 0
        
        # Initialize the embedding model
        self.embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        
        # Initialize or load the vector store
        self.vector_store = self.initialize_vector_store()
        
    def initialize_vector_store(self):
        # Check if vector store exists
        if os.path.exists("employee_vector_store"):
            print("Loading existing vector store...")
            try:
                vector_store = Chroma(
                    persist_directory="employee_vector_store",
                    embedding_function=self.embedding_model
                )
                # Verify the vector store has documents
                if vector_store._collection.count() == 0:
                    print("Vector store is empty, creating new one...")
                    return self._create_new_vector_store()
                return vector_store
            except Exception as e:
                print(f"Error loading vector store: {e}")
                return self._create_new_vector_store()
        
        print("Creating new vector store...")
        return self._create_new_vector_store()
    
    def _create_new_vector_store(self):
        # Load and process data
        jira_data, git_data, email_data = self.load_data('git-jira-email-data.json')
        self.employee_data = self.cluster_data(jira_data, git_data, email_data)
        self.employee_clusters = self.employee_data['employee_clusters']
        self.analysis = self.employee_data['analysis']
        self.collaboration_networks = self.employee_data['collaboration_networks']
        self.employee_skills = self.employee_data['employee_skills']
        
        # Create chunks
        chunks = self.chunking(self.employee_clusters)
        
        # Create and persist vector store
        vector_store = Chroma.from_texts(
            texts=chunks,
            embedding=self.embedding_model,
            persist_directory="employee_vector_store"
        )
        vector_store.persist()
        print(f"Created vector store with {len(chunks)} chunks")
        return vector_store

    def load_data(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)

        jira_data = data['jira']
        git_data = data['git']
        email_data = data['emails']

        return jira_data, git_data, email_data
    
    def get_embedding_model(self):
        try:
            with tqdm(total=100, desc="Loading model", unit="it") as pbar:
                model = SentenceTransformer('all-MiniLM-L6-v2')
                pbar.update(100)
        except Exception as e:
            print(e)
        return model
    
    def cluster_data(self, jira_data, git_data, email_data):
        try:
            clusters = process_all_employee_data(jira_data, git_data, email_data)
        except Exception as e:
            print(e)
        return clusters

    def chunking(self, clusters, chunk_size=500, overlap=200):
        # Initialize the text splitter with specified parameters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        chunks = []
        # Process each employee data with tqdm for progress tracking
        for employee_data in tqdm(clusters, desc="Chunking employee data"):
            text = str(employee_data)
            # Split the text into chunks
            text_chunks = text_splitter.split_text(text)
            chunks.extend(text_chunks)
        
        print(f"Created {len(chunks)} chunks from {len(clusters)} employee records")
        return chunks
    
    def store_in_memory(self, chunks, model):
        try:
            # Generate embeddings
            embeddings = model.encode(chunks)
            # Store chunks and embeddings in a list of tuples
            return list(zip(chunks, embeddings))
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return []
        
    def retrieve_relevant_chunks(self, query, top_k=3):
        try:
            # Use the vector store to find similar documents
            docs = self.vector_store.similarity_search(
                query,
                k=top_k,  # Only return chunks with similarity score above 0.5
            )
            
            if not docs:
                print("No relevant chunks found with similarity threshold")
                # Try again with a lower threshold
                docs = self.vector_store.similarity_search(query, k=top_k)
            
            chunks = [doc.page_content for doc in docs]
            print(f"Retrieved {len(chunks)} relevant chunks")
            return chunks
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            return []

    def query_gemini(self, query, context):
        # Gemini API endpoint and authentication 
        api_url = "https://api.groq.com/openai/v1/chat/completions"
        api_key = os.getenv("PILA_GROQ_KEY")
        
        # Create the prompt with template
        hr_analysis_template = f"""
        System: You are an HR Analytics Assistant that helps analyze employee performance data. When responding to queries:

        1. Provide concise, insightful analysis based only on the context
        2. Include relevant details like Jira IDs, commit IDs, or email references when directly relevant to the query
        3. Present performance metrics and trends clearly
        4. Maintain professional tone while highlighting achievements and areas for improvement
        5. Use bullet points for clarity when appropriate
        6. Keep sensitive personnel matters confidential

        Human Query: {query}

        Context:
        {context}
"""

        message = [{"role":"user", "content": hr_analysis_template}]
        # Set headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": message,
                "temperature": 0,
            }
        
        # Make the API request
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        
        # Process the response
        if response.status_code == 200:
            return response.json()
        else:
            return f"Error: {response.status_code}, {response.text}"

    def invoke(self, query="how is alice's performance? going on"):
        if query:
            self.relevant_chunks = self.retrieve_relevant_chunks(query)
            if not self.relevant_chunks:
                print("Warning: No relevant chunks found for the query")
            response = self.query_gemini(query, self.relevant_chunks)
            return response
        else:
            return "Please enter a query"

if __name__ == "__main__":
    bot = EmployeeDBBot()
    response = bot.invoke("how is alice's performance? going on")
    print(response['choices'][0]['message']['content'])
    print("-"*100)
    print(response['usage'])
    print("-"*100)
    print("Relevant Chunks:")
    for i, chunk in enumerate(bot.relevant_chunks, 1):
        print(f"\nChunk {i}:")
        print(chunk)
        print("-"*50)
