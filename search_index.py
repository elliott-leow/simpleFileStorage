# search_index.py
import os
import pickle
import time
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pypdf # Renamed from PyPDF2
import json

# --- Configuration ---
MODEL_NAME = 'all-MiniLM-L6-v2' # Relatively small and fast, good quality
INDEX_FILE = 'semantic_index.pkl' # Where to save the index
CACHE_DIR = os.path.expanduser("~/.cache/filebrowser_cache") # Cache directory
SUPPORTED_EXTENSIONS = ['.txt', '.pdf'] # Add more as needed
MAX_CHUNK_SIZE = 500 # Max words per chunk for embedding (adjust based on model limits/performance)
MAX_FILE_SIZE_MB = 50 # Skip files larger than this to avoid memory issues (adjust)

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)
INDEX_FILE_PATH = os.path.join(CACHE_DIR, INDEX_FILE)

# --- Global Model Loading ---
# Load the model only once
print(f"Loading sentence transformer model: {MODEL_NAME}...")
try:
    # Try CUDA if available, fall back to CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"Model loaded successfully on device: {device}")
except Exception as e:
    print(f"Error loading Sentence Transformer model: {e}")
    print("Semantic search functionality will be disabled.")
    model = None

def extract_text_from_file(filepath):
    """Extracts text content from supported file types."""
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()

    # Check file size
    try:
        if os.path.getsize(filepath) > MAX_FILE_SIZE_MB * 1024 * 1024:
             print(f"Skipping large file (>{MAX_FILE_SIZE_MB}MB): {filepath}")
             return None
    except OSError:
        return None # File might have been deleted between listing and reading

    try:
        if ext == '.txt':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        elif ext == '.pdf':
            text = ""
            try:
                reader = pypdf.PdfReader(filepath)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            except pypdf.errors.PdfReadError as pdf_err:
                 print(f"Warning: Could not read PDF {filepath}: {pdf_err}")
                 return None # Skip broken PDFs
            return text
        # Add more extractors here (e.g., python-docx for .docx)
        else:
            return None
    except Exception as e:
        print(f"Error extracting text from {filepath}: {e}")
        return None

def chunk_text(text, max_words):
    """Splits text into chunks of roughly max_words size."""
    words = text.split()
    chunks = []
    current_chunk = []
    word_count = 0
    for word in words:
        current_chunk.append(word)
        word_count += 1
        if word_count >= max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            word_count = 0
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def build_index(public_dir):
    """Scans the public directory, extracts text, creates embeddings, and saves the index."""
    if not model:
        print("Model not loaded. Cannot build index.")
        return None

    print("Starting semantic index build...")
    start_time = time.time()
    index_data = {'embeddings': [], 'metadata': []} # metadata: {'path': rel_path, 'chunk_index': i}
    files_processed = 0
    chunks_processed = 0

    for root, _, files in os.walk(public_dir):
        for filename in files:
            _, ext = os.path.splitext(filename)
            if ext.lower() not in SUPPORTED_EXTENSIONS:
                continue

            abs_path = os.path.join(root, filename)
            # Safety check (redundant if walk starts within public_dir, but good practice)
            if not abs_path.startswith(public_dir):
                continue

            rel_path = os.path.relpath(abs_path, public_dir)
            print(f"  Processing: {rel_path}")
            files_processed += 1

            text = extract_text_from_file(abs_path)
            if not text:
                continue

            chunks = chunk_text(text, MAX_CHUNK_SIZE)
            if not chunks:
                continue

            try:
                # Encode chunks in batches for efficiency
                chunk_embeddings = model.encode(chunks, convert_to_tensor=True, show_progress_bar=False)

                # Store embeddings (move to CPU if they were on GPU) and metadata
                index_data['embeddings'].append(chunk_embeddings.cpu().numpy())
                for i in range(len(chunks)):
                    index_data['metadata'].append({'path': rel_path, 'chunk_index': i})
                    chunks_processed += 1

            except Exception as encode_err:
                 print(f"Error encoding chunks for {rel_path}: {encode_err}")


    if not index_data['embeddings']:
         print("No embeddings generated. Index is empty.")
         # Create dummy structure to avoid errors later
         index_data['embeddings'] = np.array([]).reshape(0, model.get_sentence_embedding_dimension())
         index_data['metadata'] = []
    else:
        # Concatenate all embeddings into a single numpy array
        index_data['embeddings'] = np.concatenate(index_data['embeddings'], axis=0)

    try:
        with open(INDEX_FILE_PATH, 'wb') as f:
            pickle.dump(index_data, f)
        end_time = time.time()
        print(f"Semantic index built and saved to {INDEX_FILE_PATH}")
        print(f"Processed {files_processed} files, {chunks_processed} text chunks.")
        print(f"Index build took {end_time - start_time:.2f} seconds.")
        return index_data
    except Exception as e:
        print(f"Error saving index file {INDEX_FILE_PATH}: {e}")
        return None


def load_index():
    """Loads the semantic index from the file."""
    if not model:
        print("Model not loaded. Cannot load index.")
        return None
    if os.path.exists(INDEX_FILE_PATH):
        try:
            with open(INDEX_FILE_PATH, 'rb') as f:
                print(f"Loading semantic index from {INDEX_FILE_PATH}...")
                index_data = pickle.load(f)
                # Basic validation
                if isinstance(index_data, dict) and 'embeddings' in index_data and 'metadata' in index_data:
                     # Ensure embeddings is a numpy array
                     if not isinstance(index_data['embeddings'], np.ndarray):
                          print("Warning: Embeddings in index file are not a numpy array. Rebuilding recommended.")
                          return None
                     # Check if embedding dimension matches the current model
                     if index_data['embeddings'].shape[0] > 0 and index_data['embeddings'].shape[1] != model.get_sentence_embedding_dimension():
                          print(f"Warning: Index embedding dimension ({index_data['embeddings'].shape[1]}) "
                                f"does not match model dimension ({model.get_sentence_embedding_dimension()}). Rebuilding required.")
                          return None # Force rebuild if dimensions mismatch

                     print(f"Loaded index with {index_data['embeddings'].shape[0]} embeddings.")
                     return index_data
                else:
                    print("Invalid index file format. Rebuilding required.")
                    return None
        except Exception as e:
            print(f"Error loading index file {INDEX_FILE_PATH}: {e}. Rebuilding required.")
            return None
    else:
        print("Semantic index file not found.")
        return None

def semantic_search(query, index_data, top_n=15):
    """Performs semantic search using the loaded index."""
    if not model:
        print("Model not loaded. Cannot perform semantic search.")
        return []
    if index_data is None or index_data['embeddings'].shape[0] == 0:
        print("Index is not loaded or is empty. Cannot perform semantic search.")
        return []

    try:
        query_embedding = model.encode(query, convert_to_tensor=True)
        query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1) # Reshape for sklearn

        # Calculate cosine similarity
        # sims = util.pytorch_cos_sim(query_embedding, torch.tensor(index_data['embeddings']))[0]
        # Using sklearn for numpy array compatibility
        sims = cosine_similarity(query_embedding_np, index_data['embeddings'])[0]


        # Get top N results
        # Convert similarities to numpy array if using pytorch_cos_sim
        # sims_np = sims.cpu().numpy()
        sims_np = sims # Already numpy if using sklearn
        top_indices = np.argsort(sims_np)[::-1][:top_n] # Get indices of highest scores

        results = []
        seen_paths = set() # Avoid returning the same file multiple times if multiple chunks match
        for idx in top_indices:
            score = float(sims_np[idx])
            if score < 0.05: # Relevance threshold (adjust as needed)
                continue
            metadata = index_data['metadata'][idx]
            rel_path = metadata['path']
            if rel_path not in seen_paths:
                results.append({'path': rel_path, 'score': score})
                seen_paths.add(rel_path)

        # Sort final unique paths by score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

    except Exception as e:
        print(f"Error during semantic search for query '{query}': {e}")
        return []

# --- Initial Load or Build ---
# Try loading existing index first, build if needed
semantic_index = load_index()
# if semantic_index is None:
#     semantic_index = build_index(os.path.expanduser("~/public")) # Pass PUBLIC_DIR here
# Note: It's better to trigger the build explicitly (e.g., via a route)
# than doing it automatically on module import, especially if PUBLIC_DIR is large.
