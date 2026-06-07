"""
Download trained sentiment models from Google Drive
Run this script to download models to the local app directory
"""

import gdown
import os
from pathlib import Path

# Create models directory if it doesn't exist
MODEL_DIR = Path(__file__).parent / 'models'
MODEL_DIR.mkdir(exist_ok=True)

print(f"Downloading models to: {MODEL_DIR}")

# Dictionary of model_name: google_drive_file_id
# You'll need to replace these with your actual file IDs from Google Drive
MODELS = {
    'gru_best.pt': 'YOUR_GRU_FILE_ID',  # Replace with actual Google Drive file ID
    'lstm_best.pt': 'YOUR_LSTM_FILE_ID',  # Replace with actual Google Drive file ID
    'transformer_best.pt': 'YOUR_TRANSFORMER_FILE_ID',  # Replace with actual Google Drive file ID
}

def download_model(filename, file_id):
    """Download a single model from Google Drive"""
    output_path = MODEL_DIR / filename
    
    if output_path.exists():
        print(f"✓ {filename} already exists")
        return True
    
    print(f"Downloading {filename}...")
    try:
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, str(output_path), quiet=False)
        print(f"✓ Successfully downloaded {filename}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")
        return False

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Sentiment Model Downloader")
    print("="*50 + "\n")
    
    # Check if file IDs are configured
    if 'YOUR_GRU_FILE_ID' in MODELS.values():
        print("⚠️  WARNING: File IDs not configured!")
        print("\nTo use this script:")
        print("1. Go to your Google Drive")
        print("2. Right-click each model file → Get link")
        print("3. Extract the file ID from the URL (between 'id=' and '&')")
        print("4. Replace the placeholders in this script")
        print("\nExample URL: https://drive.google.com/file/d/1abc123xyz/view")
        print("File ID: 1abc123xyz")
    else:
        print("Downloading models...\n")
        success_count = 0
        for filename, file_id in MODELS.items():
            if download_model(filename, file_id):
                success_count += 1
        
        print(f"\n{'='*50}")
        print(f"Download complete: {success_count}/{len(MODELS)} models")
        print(f"{'='*50}\n")
