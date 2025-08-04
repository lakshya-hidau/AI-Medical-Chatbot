from langchain.embeddings import HuggingFaceEmbeddings

def download_hugging_face_embeddings():
    """
    Loads the 'all-MiniLM-L6-v2' embedding model using HuggingFaceEmbeddings.
    Make sure the model is cached for later use.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

# Test it
if __name__ == "__main__":
    try:
        embeddings = download_hugging_face_embeddings()
        print("✅ Embedding model loaded successfully!")
        # Example usage:
        text = "Hello, how are you?"
        vector = embeddings.embed_query(text)
        print("Vector dimension:", len(vector))
    except Exception as e:
        print("❌ Failed to load embeddings:", str(e))
