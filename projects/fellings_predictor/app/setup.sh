#!/bin/bash
# Quick Setup Script for Sentiment Prediction App

echo "🚀 Setting up Sentiment Prediction App..."

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create models directory
echo "📁 Creating models directory..."
mkdir -p models

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next Steps:"
echo "1. Download models from Google Drive (gru_best.pt, lstm_best.pt, transformer_best.pt, vocab.json)"
echo "2. Place them in the 'models/' directory"
echo "3. Run: python main.py"
echo ""
echo "Optional: Run 'python download_models.py' to auto-download (if you configure file IDs)"
