# Machine Learning & Deep Learning Journey 🚀

This repository contains all my practice projects, experiments, and implementations while learning **Machine Learning** and **Deep Learning** from scratch.

## 📁 Repository Structure

```
MachineLearning/
├── algorithmes/                           # Classic ML algorithms implemented from scratch
│   ├── bfs_problems.py                    # Graph traversal algorithms
│   ├── decision tree & random forest/     # Tree-based models
│   │   ├── Dis&Rf_usingRealData.ipynb
│   │   └── DisTree&RandForst.ipynb
│   ├── k-means & kNN/                     # Clustering & Classification
│   │   ├── k_means.ipynb
│   │   └── knn.ipynb
│   ├── linear regression/                 # Linear regression models
│   │   ├── complex_LR.py
│   │   ├── home_prices.csv
│   │   └── linear_regression.ipynb
│   ├── logistic regression/               # Classification
│   │   ├── logistic_regression.py
│   │   ├── matplotlib_code.py
│   │   └── (notebooks)
│   ├── pca/                               # Dimensionality reduction
│   │   └── pca.ipynb
│   └── SVM/                               # Support Vector Machines
│       ├── svm.ipynb
│       └── svm_kernls.py
│
├── Data_science/                          # Data analysis & exploration
│   ├── exercice.ipynb
│   ├── netflix_analitics.ipynb
│   ├── practice_exam.ipynb
│   └── tvsubscriptions.csv
│
├── models/                                # Saved trained models & competitions
│   ├── CustomerChurnPredictionforTelecommunications/
│   │   ├── model.ipynb
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   ├── Extended_Employee_Performance_and_Productivity_Data/
│   │   ├── Extended_Employee_Performance_and_Productivity_Data.csv
│   │   └── model.ipynb
│   ├── get_into_titanic/                 # Kaggle Titanic competition
│   │   ├── titanic_model.ipynb
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── gender_submission.csv
│   │   └── submission.csv
│   ├── neural_networks/                  # Deep learning models
│   │   ├── CNNs.ipynb
│   │   ├── neural_network_from_scratch.ipynb
│   │   ├── neural_netwrok.ipynb
│   │   └── rnn/
│   │       └── RNNs.ipynb
│   └── predicted-house/                  # House price prediction
│       ├── predictedhouseprice.ipynb
│       └── ParisHousing.csv
│
└── projects/                              # Complete end-to-end projects
    ├── drawing_recognition/               # Hand-drawn digit recognition app
    │   ├── README.md
    │   ├── app/
    │   │   ├── main.py
    │   │   ├── canvas_widget.py
    │   │   └── predictor.py
    │   └── model/
    │       ├── model.ipynb
    │       └── quickdraw_model.keras
    ├── fellings_predictor/                # Sentiment/feelings prediction from text
    │   ├── README.md
    │   ├── QUICKSTART.md
    │   ├── requirements.txt
    │   ├── setup.sh
    │   ├── app/
    │   │   ├── main.py
    │   │   ├── text_widget.py
    │   │   ├── predictor.py
    │   │   └── download_models.py
    │   └── model/
    │       ├── training.ipynb
    │       ├── data_exploration.ipynb
    │       ├── lstm_best.pt
    │       ├── gru_best.pt
    │       └── transformer_best.pt
    └── Minesweeper/                       # AI-powered Minesweeper game
        ├── README.md
        ├── main.py
        ├── ai_player.py
        ├── ui_main.py
        ├── button.py
        ├── settings.py
        ├── sprites.py
        └── assets/
```


## 🛠️ Tech Stack & Libraries

- **Python** 3.8+
- **ML Frameworks**: scikit-learn, TensorFlow, Keras, PyTorch
- **Data Processing**: pandas, numpy, scipy
- **Visualization**: matplotlib, seaborn
- **Deployment**: Streamlit, Flask
- **Development**: Jupyter Notebooks, VS Code
- **Utilities**: OpenCV (for drawing recognition)

## 🚀 Projects Highlight

### 1. Hand-Drawn Digit & Doodle Recognition Web App
A fully interactive web application that recognizes hand-drawn digits and doodles in real-time using a CNN trained on Google's Quick, Draw! dataset.
- **Location**: `projects/drawing_recognition/`
- **Run**: `python app/main.py`
- **Features**: 
  - Canvas widget for real-time drawing
  - Instant predictions
  - Model: `quickdraw_model.keras`

### 2. Feelings/Sentiment Predictor
An NLP project predicting sentiments/feelings from text using multiple deep learning architectures.
- **Location**: `projects/fellings_predictor/`
- **Models**: LSTM, GRU, Transformer
- **Run**: `bash setup.sh` then `python app/main.py`
- **See**: `app/QUICKSTART.md` for quick start guide

### 3. AI-Powered Minesweeper
A classic Minesweeper game with an intelligent AI player that can solve boards automatically.
- **Location**: `projects/Minesweeper/`
- **Run**: `python main.py`
- **Features**: 
  - Play manually or watch AI solve
  - Adjustable difficulty & board size
  - Real-time AI strategy visualization

## 📊 Model Competitions & Datasets

- **Titanic Survival Prediction** (`models/get_into_titanic/`) - Kaggle competition
- **Telecommunication Customer Churn** (`models/CustomerChurnPredictionforTelecommunications/`)
- **Employee Performance Analytics** (`models/Extended_Employee_Performance_and_Productivity_Data/`)
- **House Price Prediction** (`models/predicted-house/`) - Paris Housing dataset

## 🧠 Algorithm Implementations

From-scratch implementations of fundamental ML algorithms:
- **Classification**: Logistic Regression, SVM with kernels, Decision Trees, Random Forest
- **Clustering**: K-Means
- **Dimensionality Reduction**: PCA
- **Regression**: Linear Regression (simple & complex), Polynomial Regression
- **Graph Algorithms**: BFS for problem-solving
- **Neural Networks**: From scratch implementations, CNNs, RNNs

## 📈 Learning Path

1. **Classic Algorithms** → `algorithmes/` folder
2. **Statistical Models** → `models/` folder (Kaggle competitions)
3. **Deep Learning** → `models/neural_networks/` (CNNs, RNNs)
4. **Production Projects** → `projects/` folder (end-to-end deployable apps)

## 🔧 Getting Started

### Setup

```bash
# Clone or explore the repository
cd MachineLearning

# For individual projects:
cd projects/drawing_recognition
python app/main.py

# For deep learning models:
cd projects/fellings_predictor
bash setup.sh
python app/main.py
```

### Requirements
Most notebooks and scripts use standard ML libraries. For specific projects:
- **Projects**: See `requirements.txt` in each project folder
- **Notebooks**: Run in Jupyter with essential libraries installed

## 💡 Philosophy
This is a personal learning playground where I practice and experiment with ML/DL concepts. Code is honest, well-commented, and continuously improving. Concepts are learned from courses, papers, and hands-on experimentation.

## 🎯 Use Cases for This Repo
- Learning ML algorithms from scratch
- Understanding deep learning architectures (CNNs, RNNs, Transformers)
- Studying Kaggle competition solutions
- Reference for deploying ML projects
- Practice implementing classic algorithms

## 📚 Resources Used
- Fast.ai courses
- Kaggle datasets & competitions
- TensorFlow & PyTorch official documentation
- Scikit-learn tutorials
- Research papers & blog posts

---

Made with ❤️ and a lot of coffee ☕ by Zakary
