from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# print(PINECONE_API_KEY)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as LangchainPinecone
from tqdm import tqdm

# 1. Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# 2. Generate one sample embedding to detect correct dimension
sample_vector = embeddings.embed_query("Hello world")
dimension = len(sample_vector)  # e.g., 384 if using all-MiniLM-L6-v2
print(f"📏 Detected embedding dimension: {dimension}")

# 3. Index Configuration
index_name = "medical-chatbot"

# 4. Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    print("🆕 Creating Pinecone index...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# 5. Connect to the index
index = pc.Index(index_name)

# 6. Convert text chunks to plain string list
texts = [chunk.page_content for chunk in text_chunks]
print("✅ Total Chunks:", len(texts))

# 7. Create vector objects
vectors = []
for i, text in enumerate(texts):
    try:
        vector = embeddings.embed_query(text)
        if len(vector) != dimension:
            raise ValueError(f"❌ Mismatch: Expected {dimension}, got {len(vector)}")
        vectors.append({
            "id": f"chunk-{i}",
            "values": vector,
            "metadata": {"text": text}
        })
    except Exception as e:
        print(f"⚠️ Skipping chunk {i} due to error: {e}")

# 8. Upload in safe batches (20–100 max)
batch_size = 20
print("📤 Uploading vectors to Pinecone in batches...")

for i in tqdm(range(0, len(vectors), batch_size)):
    batch = vectors[i:i + batch_size]
    try:
        index.upsert(vectors=batch)
    except Exception as e:
        print(f"❌ Error uploading batch {i//batch_size + 1}: {e}")

print("✅ All valid vectors uploaded successfully!")
