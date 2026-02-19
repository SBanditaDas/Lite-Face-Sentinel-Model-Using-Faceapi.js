"""
Download Pre-built Face Recognition Model
Downloads ready-to-use TensorFlow.js face recognition model from face-api.js

This downloads the smallest, optimized model for your needs.
"""

import os
import urllib.request
import json

# Model URLs from face-api.js repository
BASE_URL = "https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights"

# We'll use the tiny face detector + tiny landmarks for smallest size
MODELS = {
    "tiny_face_detector": [
        "tiny_face_detector_model-weights_manifest.json",
        "tiny_face_detector_model-shard1",
    ],
    "face_landmark_68_tiny": [
        "face_landmark_68_tiny_model-weights_manifest.json",
        "face_landmark_68_tiny_model-shard1",
    ],
    "face_recognition": [
        "face_recognition_model-weights_manifest.json", 
        "face_recognition_model-shard1",
        "face_recognition_model-shard2",
    ]
}

OUTPUT_DIR = "face_recognition_model"

def download_file(url, output_path):
    """Download a file with progress"""
    print(f"  Downloading {os.path.basename(output_path)}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        size = os.path.getsize(output_path) / 1024
        print(f"    ✓ Downloaded ({size:.1f} KB)")
        return size
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return 0

def main():
    print("=" * 70)
    print("FACE RECOGNITION MODEL DOWNLOADER")
    print("Downloading pre-built TensorFlow.js models from face-api.js")
    print("=" * 70)
    print()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_size = 0
    
    for model_name, files in MODELS.items():
        print(f"\nDownloading {model_name}...")
        for filename in files:
            url = f"{BASE_URL}/{filename}"
            output_path = os.path.join(OUTPUT_DIR, filename)
            size = download_file(url, output_path)
            total_size += size
    
    print("\n" + "=" * 70)
    print(f"TOTAL MODEL SIZE: {total_size / 1024:.2f} MB")
    print("=" * 70)
    
    if total_size / 1024 <= 8.0:
        print("✅ SUCCESS! Model is under 8 MB!")
    
    print(f"\n✓ Downloaded to: {OUTPUT_DIR}/")
    print()

if __name__ == "__main__":
    main()
