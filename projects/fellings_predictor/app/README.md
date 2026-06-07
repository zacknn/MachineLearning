# Sentiment Prediction App 🎯

A tkinter-based GUI application that performs sentiment analysis using 3 different neural network architectures: **GRU**, **LSTM**, and **Transformer**.

## Features

- 📝 **Text Input**: Clean interface to enter text for analysis
- 🤖 **3 Models**: Predictions from GRU, LSTM, and Transformer models
- 📊 **Visual Results**: Shows sentiment, confidence, and probability distribution for each model
- 🎨 **Beautiful UI**: Color-coded results (Green=Positive, Orange=Neutral, Red=Negative)

## Setup Instructions

### 1. Prerequisites

```bash
pip install torch torchvision torchaudio
pip install gdown
```

### 2. Download Models

#### Option A: Automatic Download
```bash
cd fellings_predictor/app
python download_models.py
```

Edit `download_models.py` first and add your Google Drive file IDs.

#### Option B: Manual Download
1. Go to your Google Drive (`sentiment_project` folder)
2. Download these files:
   - `gru_best.pt` → `app/models/gru_best.pt`
   - `lstm_best.pt` → `app/models/lstm_best.pt`
   - `transformer_best.pt` → `app/models/transformer_best.pt`
   - `vocab.json` → `app/models/vocab.json`

3. Create the `models` directory if it doesn't exist:
```bash
mkdir -p fellings_predictor/app/models
```

### 3. Run the App

```bash
cd fellings_predictor/app
python main.py
```

## File Structure

```
fellings_predictor/app/
├── main.py                 # Main tkinter application
├── predictor.py            # Model loading and prediction logic
├── text_widget.py          # Text input widget
├── download_models.py      # Script to download models from Google Drive
└── models/                 # Downloaded models directory
    ├── gru_best.pt
    ├── lstm_best.pt
    ├── transformer_best.pt
    └── vocab.json
```

## How to Use

1. **Enter Text**: Type or paste text in the input area
2. **Click Predict**: Press the "🔮 Predict" button
3. **View Results**: See predictions from all 3 models with:
   - **Sentiment Label**: Negative, Neutral, or Positive
   - **Confidence**: Percentage confidence in the prediction
   - **Probability Distribution**: Breakdown of all sentiment scores
4. **Clear**: Use the "🗑️ Clear" button to reset

## Model Architectures

| Model | Embedding | Hidden | Layers | Dropout |
|-------|-----------|--------|--------|---------|
| GRU | 128 | 256 | 2 | 0.2 |
| LSTM | 128 | 256 | 2 | 0.2 |
| Transformer | 64 | 256 (FF) | 2 | 0.1 |

## Training Information

See `../model/training.ipynb` for:
- Data preprocessing
- Model architecture definitions
- Training procedures
- Hyperparameter configurations

## Troubleshooting

**Error: "Model not found"**
- Check that model files exist in `app/models/`
- Verify file names are correct (case-sensitive on Linux)

**Error: "vocab.json not found"**
- Download `vocab.json` from Google Drive or create it during training
- Ensure it's in the `app/models/` directory

**CUDA/GPU Issues**
- App forces CPU mode by default (see `os.environ['CUDA_VISIBLE_DEVICES']`)
- To enable GPU, remove that line from `main.py`

## Performance Tips

- **First Run**: Models load into memory (~200MB total)
- **Prediction Speed**: ~100-500ms per model
- **Batch Processing**: Currently processes one text at a time

## License

Educational project for sentiment analysis learning.
