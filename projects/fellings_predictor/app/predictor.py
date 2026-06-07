import torch
import torch.nn as nn
import json
import re
from pathlib import Path
import numpy as np

# Get the directory where this script is located
APP_DIR = Path(__file__).parent
MODEL_DIR = APP_DIR / 'models'

# ==================== Model Architectures ====================

class sentimentGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=False,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.embedding(x)
        output, h = self.gru(x)
        x = self.dropout(h[-1])
        x = self.fc(x)
        return x


class sentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=False,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.embedding(x)
        output, (h, c) = self.lstm(x)
        x = self.dropout(h[-1])
        x = self.fc(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() *
            (-np.log(10000.0) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SentimentTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes,
                 num_heads=4, num_layers=2, ff_dim=256, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x):
        x = self.pos_encoding(self.embedding(x))
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(self.dropout(x))
        return x


# ==================== Sentiment Predictor ====================

class SentimentPredictor:
    def __init__(self, vocab_json_path=None):
        """
        Initialize sentiment predictor with 3 models
        vocab_json_path: path to vocab.json file (optional)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Sentiment labels
        self.sentiment_labels = ['Negative', 'Neutral', 'Positive']
        
        # Load vocabulary
        if vocab_json_path and Path(vocab_json_path).exists():
            with open(vocab_json_path, 'r') as f:
                self.word2idx = json.load(f)
        else:
            # Create a basic vocab from common words if json not found
            self.word2idx = self._create_default_vocab()
        
        self.vocab_size = len(self.word2idx)
        
        # Initialize models
        self.models = {
            'GRU': self._load_gru_model(),
            'LSTM': self._load_lstm_model(),
            'Transformer': self._load_transformer_model()
        }
    
    def _create_default_vocab(self):
        """Create a default vocab if json not available"""
        vocab = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
        common_words = ['the', 'a', 'is', 'good', 'bad', 'great', 'awful', 'nice', 
                       'hate', 'love', 'amazing', 'terrible', 'excellent', 'horrible']
        for i, word in enumerate(common_words, start=4):
            vocab[word] = i
        return vocab
    
    def _load_gru_model(self):
        """Load GRU model"""
        model = sentimentGRU(
            vocab_size=self.vocab_size,
            embedding_dim=128,
            hidden_dim=256,
            output_dim=3,
            n_layers=2,
            dropout=0.2
        )
        model_path = MODEL_DIR / 'gru_best.pt'
        
        if model_path.exists():
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"✓ Loaded GRU model from {model_path}")
            except Exception as e:
                print(f"✗ Failed to load GRU model: {e}")
        else:
            print(f"⚠️  GRU model not found at {model_path}")
        
        model.to(self.device)
        model.eval()
        return model
    
    def _load_lstm_model(self):
        """Load LSTM model"""
        model = sentimentLSTM(
            vocab_size=self.vocab_size,
            embedding_dim=128,
            hidden_dim=256,
            output_dim=3,
            n_layers=2,
            dropout=0.2
        )
        model_path = MODEL_DIR / 'lstm_best.pt'
        
        if model_path.exists():
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"✓ Loaded LSTM model from {model_path}")
            except Exception as e:
                print(f"✗ Failed to load LSTM model: {e}")
        else:
            print(f"⚠️  LSTM model not found at {model_path}")
        
        model.to(self.device)
        model.eval()
        return model
    
    def _load_transformer_model(self):
        """Load Transformer model"""
        model = SentimentTransformer(
            vocab_size=self.vocab_size,
            embedding_dim=64,
            num_classes=3,
            num_heads=4,
            num_layers=2,
            ff_dim=256,
            dropout=0.1
        )
        model_path = MODEL_DIR / 'transformer_best.pt'
        
        if model_path.exists():
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"✓ Loaded Transformer model from {model_path}")
            except Exception as e:
                print(f"✗ Failed to load Transformer model: {e}")
        else:
            print(f"⚠️  Transformer model not found at {model_path}")
        
        model.to(self.device)
        model.eval()
        return model
    
    def preprocess_text(self, text, max_length=50):
        """Convert text to token indices"""
        # Basic preprocessing
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)  # Remove special chars
        
        tokens = text.split()[:max_length]
        tokens = ['<START>'] + tokens + ['<END>']
        
        indices = [self.word2idx.get(t, self.word2idx['<UNK>']) for t in tokens]
        return torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(self.device)
    
    def predict(self, text):
        """
        Predict sentiment for given text
        Returns: {model_name: {label, confidence}}
        """
        inputs = self.preprocess_text(text)
        results = {}
        
        with torch.no_grad():
            for model_name, model in self.models.items():
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)[0]
                
                pred_idx = probabilities.argmax().item()
                confidence = probabilities[pred_idx].item()
                
                results[model_name] = {
                    'sentiment': self.sentiment_labels[pred_idx],
                    'confidence': confidence,
                    'probabilities': {
                        'Negative': float(probabilities[0]),
                        'Neutral': float(probabilities[1]),
                        'Positive': float(probabilities[2])
                    }
                }
        
        return results


# Initialize global predictor (lazy loaded)
_predictor = None

def get_predictor():
    """Get or initialize the predictor"""
    global _predictor
    if _predictor is None:
        _predictor = SentimentPredictor()
    return _predictor

def predict_sentiment(text):
    """Predict sentiment for text"""
    predictor = get_predictor()
    return predictor.predict(text)
