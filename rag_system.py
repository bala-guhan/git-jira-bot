import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to chunk Python file
def chunk_python_file(file_path, chunk_size=500):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Split content into chunks based on character count
        chunks = []
        current_chunk = ""
        for line in content.split('\n'):
            if len(current_chunk) + len(line) < chunk_size:
                current_chunk += line + '\n'
            else:
                chunks.append(current_chunk.strip())
                current_chunk = line + '\n'
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

# Function to store chunks and embeddings in memory
def store_in_memory(chunks):
    try:
        # Generate embeddings
        embeddings = model.encode(chunks)
        # Store chunks and embeddings in a list of tuples
        return list(zip(chunks, embeddings))
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []

# Function to retrieve relevant chunks for a query
def retrieve_relevant_chunks(query, stored_data, top_k=3):
    try:
        # Get query embedding
        query_embedding = model.encode([query])[0]
        
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

# Main function to process file and handle queries
def main(file_path, query):
    # Chunk the Python file
    chunks = chunk_python_file(file_path)
    if not chunks:
        print("No chunks created. Exiting.")
        return
    
    # Store chunks and embeddings in memory
    stored_data = store_in_memory(chunks)
    if not stored_data:
        print("No data stored. Exiting.")
        return
    
    print("Chunks and embeddings stored in memory.")
    
    # Process query
    if query:
        relevant_chunks = retrieve_relevant_chunks(query, stored_data)
        print("\nRelevant chunks for query:")
        for i, chunk in enumerate(relevant_chunks, 1):
            print(f"\nChunk {i}:\n{chunk}")

# Example usage
if __name__ == "__main__":
    # Replace with your Python file path and query
    sample_file = "sample.py"
    sample_query = "How to define a function in Python"
    
    # Create a sample Python file for testing
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write('''
def greet(name):
    return f"Hello, {name}!"

class Example:
    def __init__(self):
        self.value = 0
    
    def increment(self):
        self.value += 1

if __name__ == "__main__":
    obj = Example()
    obj.increment()
    print(greet("World"))
        ''')
    
    main(sample_file, sample_query)