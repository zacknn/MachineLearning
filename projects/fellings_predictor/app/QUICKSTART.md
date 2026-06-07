# Quick Start Guide 🚀

## Step 1: Download Models from Google Drive

Your trained models are saved in your Google Drive under `sentiment_project/`:
- `gru_best.pt`
- `lstm_best.pt`
- `transformer_best.pt`
- `vocab.json`

**Download all 4 files and save them to:**
```
fellings_predictor/app/models/
```

Create the `models` folder if it doesn't exist.

## Step 2: Install Dependencies

```bash
cd fellings_predictor/app
pip install -r requirements.txt
```

Or manually:
```bash
pip install torch gdown
```

## Step 3: Run the App

```bash
python main.py
```

The GUI will open with:
- Text input area at the top
- Predict and Clear buttons
- Three result panels (one for each model)

## Step 4: Use the App

1. **Type or paste text** in the input area
2. **Click "Predict"** button
3. **See results** from all 3 models showing:
   - ✅ Sentiment (Positive/Neutral/Negative)
   - 📊 Confidence percentage
   - 📈 Probability distribution

## Example Input

Try these sample texts:
- "I love this product! It's amazing!" (should be Positive)
- "This is okay, nothing special" (should be Neutral)
- "This is terrible and I hate it" (should be Negative)

## Troubleshooting

### Models not loading
- Check that model files exist in `app/models/`
- Check file names are correct

### App crashes on predict
- Make sure `vocab.json` is in `app/models/`
- Check that all 4 files are in the models folder

### First run is slow
- Models are loading into memory (~200MB)
- Subsequent predictions will be faster

## File Structure

```
fellings_predictor/
├── model/
│   ├── training.ipynb        ← Training code
│   └── ...
└── app/
    ├── main.py               ← Run this! 🎯
    ├── predictor.py          ← Model logic
    ├── text_widget.py        ← Text input
    ├── download_models.py    ← Download script
    ├── models/               ← Put models here
    │   ├── gru_best.pt
    │   ├── lstm_best.pt
    │   ├── transformer_best.pt
    │   └── vocab.json
    ├── README.md
    └── requirements.txt
```

## Tips for Better Predictions

1. **Clean text**: Remove special characters and URLs
2. **Longer text**: More context helps models predict better
3. **Similar domain**: Models work best on similar sentiment data
4. **Compare results**: Check predictions from all 3 models

Happy predicting! 🎉
