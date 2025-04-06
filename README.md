# Multimodal Video Embedding for RAG

A Python application for embedding video content into vector databases for Retrieval-Augmented Generation (RAG) applications.

## Overview

This project demonstrates how to create multimodal embeddings from video content by extracting and combining visual and audio features. These embeddings are then stored in a vector database (ChromaDB) to enable semantic search and retrieval of specific video segments.

## Features

- Extract frames from video files at configurable intervals
- Transcribe audio from videos using Whisper
- Generate visual embeddings using CLIP
- Create text embeddings from transcriptions
- Fuse modalities into unified vector representations
- Store multimodal embeddings in ChromaDB with proper timestamps
- Perform semantic search to find relevant video segments

## Requirements

- Python >= 3.12
- uv package manager

## Installation

### 1. Install uv

For macOS and Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For Windows:

```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone the repository:

```bash
git clone https://github.com/yourusername/multimodal-video-embedding.git
cd multimodal-video-embedding
```

### 3. Create virtual environment and install dependencies:

```bash
# Create a virtual environment
uv venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

## Usage

### Running with uv

```bash
# Run the main script
uv run embeddings.py

# Run with specific Python version
uv run --python=3.10 src/main.py
```

## Development

### Adding new dependencies

```bash
# Add a dependency
uv pip add torch transformers

# Add development dependencies
uv pip add --dev pytest black

# Generate or update requirements.txt
uv pip compile pyproject.toml --output-file requirements.txt
```

### Using uv's project management features (experimental)

```bash
# Initialize a new project (alternative to pyproject.toml)
uv project init

# Add dependencies to project
uv project add torch transformers pytube

# Run with automatic dependency installation
uv run src/main.py
```

## How It Works

### Multimodal Feature Extraction

The system extracts features from multiple modalities:
1. **Visual**: Frames are sampled from the video at regular intervals
2. **Audio**: Speech is transcribed to text with timestamps

### Embedding Generation

1. **Visual Embeddings**: Generated using CLIP visual encoder
2. **Text Embeddings**: Generated using Sentence Transformers
3. **Fusion**: Visual and text embeddings are aligned by timestamp and combined

### Vector Database Integration

Custom embedding function implementation for ChromaDB enables:
1. Storage of pre-computed fusion embeddings
2. Query handling for semantic search
3. Temporal alignment of results with video timestamps

## Limitations and Future Work

- The custom embedding function has limited query embedding generation capabilities
- Production implementations should enhance the `__call__` method to generate proper multimodal embeddings on-the-fly
- Performance optimizations needed for larger video collections
- Integration with streaming video sources could improve real-time capabilities

## License

MIT

## Acknowledgments

- OpenAI for CLIP and Whisper models
- The ChromaDB team for their vector database implementation
- Sentence Transformers for text embedding models
- Astral for the uv package manager