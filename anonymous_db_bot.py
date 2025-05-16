import os
import json
import requests
from dotenv import load_dotenv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from cluster_knowledge import cluster_task_data
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

class AnonymousDBBot:
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
        jira_data, git_data, email_data = self.load_data('git-jira-email-data.json')
        self.clusters = self.cluster_data(jira_data, git_data, email_data)
        self.chunks = self.chunking(self.clusters)
        self.model = self.get_embedding_model()
        self.stored_data = self.store_in_memory(self.chunks, self.model)

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
            clusters = cluster_task_data(jira_data, git_data, email_data)
        except Exception as e:
            print(e)
        return clusters

    def chunking(self, clusters):
        """Project wise chunking"""
        chunks = []
        for item in clusters:
            chunks.append(str(item))
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
        
    def retrieve_relevant_chunks(query, stored_data, model, top_k=3):
        try:
            # Get query embedding
            query_embedding = model.encode(query)
            
            if not stored_data:
                return []
            
            # Calculate cosine similarities
            similarities = []
            for chunk, embedding in stored_data:
                similarity = cosine_similarity([query_embedding], [embedding])[0][0]
                similarities.append((similarity, chunk))
            
            # Sort by similarity and get top_k
            similarities.sort(reverse=True)
            return [chunk for _, chunk in similarities[:top_k]]
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            return []

    def query_gemini(self, query, context):
        # Gemini API endpoint and authentication 
        api_url = "https://api.groq.com/openai/v1/chat/completions"  # Replace with actual Gemini endpoint
        api_key = os.getenv("PILA_GROQ_KEY")
        
        # Create the prompt with template
        prompt = f"""
        System: You are a technical documentation assistant. Focus exclusively on explaining the code changes and technical solutions from the provided context.
        
        When answering:
        1. Only describe the technical issue and how it was resolved
        2. Exclude all employee names, commit IDs, ticket IDs, and email conversations
        3. Concentrate solely on what code was modified and why
        4. Explain the technical impact of these changes
        
        Human Query: {query}
        
        Context:
        {context}
        """

        message = [{"role":"user", "content": prompt}]
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

    def invoke(self, query):
        if query:
            self.relevant_chunks = self.retrieve_relevant_chunks(query, self.stored_data, self.model)
            response = self.query_gemini(query, self.relevant_chunks)
            return response
        else:
            return "Please enter a query"

if __name__ == "__main__":
    bot = AnonymousDBBot()
    response = bot.invoke("how did we solve the dashboard problem")
    print(response['choices'][0]['message']['content'])
    print("-"*100)
    print(response['usage'])
    print("-"*100)
    print("Relevant Chunks:")
    print(bot.relevant_chunks)
