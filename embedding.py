# Multimodal Video Embedding for RAG
# This example demonstrates a simplified workflow for embedding video content into a vector database

import os
import torch
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
from pytubefix import YouTube
import cv2
import numpy as np
import whisper
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
import json
from datetime import timedelta

# ===================== STEP 1: VIDEO ACQUISITION =====================
def download_video(youtube_url, output_path="video.mp4"):
    """Download a video from YouTube for processing"""
    print(f"Downloading video from {youtube_url}...")
    yt = YouTube(youtube_url)
    yt.streams.filter(progressive=True, file_extension="mp4").first().download(filename=output_path)
    print(f"Downloaded video: {yt.title}")
    return output_path, yt.title

# ===================== STEP 2: MULTIMODAL FEATURE EXTRACTION =====================
def extract_video_frames(video_path, sample_rate=1):
    """Extract frames from video at a given sample rate (1 frame per n seconds)"""
    print("Extracting frames from video...")
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    frames = []
    frame_timestamps = []
    
    sample_interval = int(fps * sample_rate)
    
    success, frame = video.read()
    count = 0
    
    while success:
        if count % sample_interval == 0:
            # Convert from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            timestamp = count / fps
            frame_timestamps.append(timestamp)
            
        success, frame = video.read()
        count += 1
    
    video.release()
    print(f"Extracted {len(frames)} frames from {duration:.2f} seconds of video")
    return frames, frame_timestamps

def extract_audio_transcript(video_path):
    """Extract and transcribe audio from video"""
    print("Transcribing audio from video...")
    # Load Whisper model
    model = whisper.load_model("base")
    
    # Transcribe audio
    result = model.transcribe(video_path)
    
    # Get segments with timestamps
    segments = result["segments"]
    
    transcripts = []
    for segment in segments:
        transcripts.append({
            "text": segment["text"],
            "start": segment["start"],
            "end": segment["end"]
        })
    
    print(f"Transcribed {len(transcripts)} audio segments")
    return transcripts

# ===================== STEP 3: EMBEDDING GENERATION =====================
def generate_visual_embeddings(frames):
    """Generate embeddings for video frames using CLIP"""
    print("Generating visual embeddings...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    embeddings = []
    batch_size = 8
    
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        inputs = processor(images=batch, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            visual_features = model.get_image_features(**inputs)
            
        embeddings.extend(visual_features.cpu().numpy())
        
    embeddings = np.array(embeddings)
    print(f"Generated {len(embeddings)} visual embeddings of dimension {embeddings.shape[1]}")
    return embeddings

def generate_text_embeddings(transcripts):
    """Generate embeddings for transcript segments"""
    print("Generating text embeddings...")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    texts = [segment["text"] for segment in transcripts]
    embeddings = []
    
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Use mean pooling to get a fixed-size embedding
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            batch_embeddings = sum_embeddings / sum_mask
            embeddings.extend(batch_embeddings.cpu().numpy())
    
    embeddings = np.array(embeddings)
    print(f"Generated {len(embeddings)} text embeddings of dimension {embeddings.shape[1]}")
    return embeddings

def fusion_embeddings(visual_embeddings, text_embeddings, frame_timestamps, transcripts):
    """Fuse visual and text embeddings through late fusion approach"""
    print("Performing multimodal fusion...")
    
    # Determine which text segment corresponds to each frame based on timestamps
    multimodal_embeddings = []
    metadata = []
    
    for i, (frame_emb, timestamp) in enumerate(zip(visual_embeddings, frame_timestamps)):
        # Find all transcript segments that overlap with this frame
        relevant_segments = []
        for j, segment in enumerate(transcripts):
            if segment["start"] <= timestamp <= segment["end"]:
                relevant_segments.append(j)
        
        if relevant_segments:
            # Simple approach: average the embeddings of relevant text segments
            relevant_text_embs = text_embeddings[relevant_segments]
            avg_text_emb = np.mean(relevant_text_embs, axis=0)
            
            # Concatenate visual and text embeddings
            # Note: In practice, you might want to normalize and/or apply weights
            fused_emb = np.concatenate([frame_emb, avg_text_emb])
            
            # Create metadata for this chunk
            segment_texts = [transcripts[j]["text"] for j in relevant_segments]
            meta = {
                "timestamp": timestamp,
                "timestamp_formatted": str(timedelta(seconds=int(timestamp))),
                "frame_index": i,
                "text": " ".join(segment_texts)
            }
            
            multimodal_embeddings.append(fused_emb)
            metadata.append(meta)
    
    print(f"Created {len(multimodal_embeddings)} multimodal embeddings")
    return np.array(multimodal_embeddings), metadata

# ===================== STEP 4: VECTOR DATABASE STORAGE =====================
# First, define a custom embedding function for our multimodal fusion embeddings

class MultimodalEmbeddingFunction(EmbeddingFunction[Documents]):
    """Custom embedding function for pre-computed multimodal embeddings.
    
    This function allows ChromaDB to use our pre-computed fusion embeddings
    during both storage and query operations.
    """
    def __init__(self, embeddings_map=None):
        """Initialize with optional pre-computed embeddings map"""
        self.embeddings_map = embeddings_map or {}  # Maps document text to embedding
        self.dimension = None  # Will be set when first embeddings are seen
    
    def __call__(self, input: Documents) -> Embeddings:
        """Return embeddings for the given input documents.
        
        For query operations, we need to compute new embeddings on the fly.
        For now, this is a placeholder that would be replaced with actual
        multimodal embedding computation in a real implementation.
        """
        if not input:
            return []
        
        # If we have pre-computed embeddings for these exact documents, use them
        result = []
        for doc in input:
            if doc in self.embeddings_map:
                result.append(self.embeddings_map[doc])
            else:
                # In a real implementation, you would compute the multimodal embedding here
                # For this example, we'll just use a placeholder that would fail in practice
                print(f"Warning: No pre-computed embedding found for: {doc[:50]}...")
                # Return a zero vector of the right dimension if we know it
                if self.dimension:
                    result.append([0.0] * self.dimension)
                else:
                    # This would fail in practice - just for example purposes
                    result.append([0.0] * 512)  # Placeholder
        
        return result
    
    def add_embeddings(self, documents, embeddings):
        """Add pre-computed embeddings to the map"""
        if len(embeddings) > 0 and not self.dimension:
            self.dimension = len(embeddings[0])
            
        for doc, emb in zip(documents, embeddings):
            self.embeddings_map[doc] = emb

def store_in_vectordb(embeddings, metadata, collection_name="video_embeddings"):
    """Store embeddings in a ChromaDB vector database"""
    print("Storing embeddings in vector database...")
    
    # Initialize ChromaDB client
    client = chromadb.Client()
    
    # Create documents list from metadata
    documents = [meta["text"] for meta in metadata]
    
    # Create and configure our custom embedding function
    embedding_func = MultimodalEmbeddingFunction()
    embedding_func.add_embeddings(documents, embeddings)
    
    # Create or get a collection with our custom embedding function
    try:
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_func,
            metadata={"hnsw:space": "cosine"}  # Using cosine similarity
        )
    except:
        # If collection already exists, get it with our embedding function
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_func
        )
    
    # Add embeddings to the collection
    ids = [f"chunk_{i}" for i in range(len(embeddings))]
    metadatas = metadata
    
    # Add embeddings in batches
    batch_size = 100
    for i in range(0, len(embeddings), batch_size):
        end_idx = min(i + batch_size, len(embeddings))
        collection.add(
            ids=ids[i:end_idx],
            embeddings=embeddings[i:end_idx].tolist(),
            metadatas=metadatas[i:end_idx],
            documents=documents[i:end_idx]
        )
    
    print(f"Stored {len(embeddings)} embeddings in ChromaDB collection '{collection_name}'")
    return collection

# ===================== STEP 5: RAG QUERYING =====================
def query_video_rag(query_text, collection, top_k=5):
    """Query the vector database to find relevant video segments"""
    print(f"Querying for: '{query_text}'")
    
    # In a real production scenario, we would:
    # 1. Process the query text through our text embedding model
    # 2. Create a multimodal embedding placeholder (with zeros for visual features)
    # 3. Use that embedding for querying
    #
    # However, our MultimodalEmbeddingFunction has limited capability to create
    # query embeddings on the fly since we're not implementing the full video
    # processing pipeline for queries.
    #
    # ChromaDB will use our embedding_function to process the query text
    # with our placeholder implementation.
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k
    )
    
    print(f"Found {len(results['ids'][0])} matches")
    
    # Format results
    formatted_results = []
    for i in range(len(results["ids"][0])):
        result = {
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "timestamp": results["metadatas"][0][i]["timestamp_formatted"],
            "similarity": results["distances"][0][i] if "distances" in results else None
        }
        formatted_results.append(result)
    
    return formatted_results

# ===================== MAIN WORKFLOW =====================
def main():
    # Example usage
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Example video
    
    # Step 1: Download video
    video_path, video_title = download_video(youtube_url)
    
    # Step 2: Extract features
    frames, frame_timestamps = extract_video_frames(video_path, sample_rate=2)  # 1 frame every 2 seconds
    transcripts = extract_audio_transcript(video_path)
    
    # Step 3: Generate embeddings
    visual_embeddings = generate_visual_embeddings(frames)
    text_embeddings = generate_text_embeddings(transcripts)
    
    # Step 4: Fuse embeddings
    multimodal_embeddings, metadata = fusion_embeddings(
        visual_embeddings, text_embeddings, frame_timestamps, transcripts
    )
    
    # Step 5: Store in vector database
    collection = store_in_vectordb(multimodal_embeddings, metadata, f"video_NeverGonnaGiveYouUp")
    
    # Step 6: Example RAG query
    results = query_video_rag("dancing in the video", collection)
    
    # Print results
    print("\nQuery Results:")
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Timestamp: {result['timestamp']}")
        print(f"  Text: {result['text']}")
        print(f"  Similarity: {result['similarity']}")
        print()

if __name__ == "__main__":
    main()