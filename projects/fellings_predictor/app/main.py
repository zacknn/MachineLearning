"""
Sentiment Prediction App using Tkinter
Displays predictions from GRU, LSTM, and Transformer models
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage

from tkinter import *
from tkinter import messagebox
from text_widget import TextInputWidget
from predictor import predict_sentiment
from pathlib import Path


class SentimentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentiment Prediction - 3 Models")
        self.root.geometry("900x700")
        self.root.configure(bg="#f5f5f5")
        
        # Title
        title_label = Label(
            root,
            text="📊 Sentiment Analysis with 3 Models",
            font=("Arial", 18, "bold"),
            bg="#f5f5f5"
        )
        title_label.pack(pady=10)
        
        # Instructions
        instructions = Label(
            root,
            text="Enter text below to get sentiment predictions from GRU, LSTM, and Transformer models",
            font=("Arial", 10),
            bg="#f5f5f5",
            fg="#666"
        )
        instructions.pack(pady=5)
        
        # Text input section
        input_frame = LabelFrame(
            root,
            text="📝 Input Text",
            font=("Arial", 11, "bold"),
            bg="#f5f5f5",
            padx=10,
            pady=10
        )
        input_frame.pack(padx=20, pady=10, fill=BOTH, expand=True)
        
        self.text_widget = TextInputWidget(input_frame, height=8, width=80)
        self.text_widget.pack(fill=BOTH, expand=True)
        
        # Buttons frame
        button_frame = Frame(root, bg="#f5f5f5")
        button_frame.pack(pady=10)
        
        Button(
            button_frame,
            text="🔮 Predict",
            command=self.predict,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 11, "bold"),
            padx=15,
            pady=8,
            cursor="hand2"
        ).pack(side=LEFT, padx=5)
        
        Button(
            button_frame,
            text="🗑️  Clear",
            command=self.clear,
            bg="#f44336",
            fg="white",
            font=("Arial", 11, "bold"),
            padx=15,
            pady=8,
            cursor="hand2"
        ).pack(side=LEFT, padx=5)
        
        # Results frame
        results_frame = LabelFrame(
            root,
            text="📈 Predictions",
            font=("Arial", 11, "bold"),
            bg="#f5f5f5",
            padx=10,
            pady=10
        )
        results_frame.pack(padx=20, pady=10, fill=BOTH, expand=True)
        
        # Create result frames for each model
        self.result_frames = {}
        models = ['GRU', 'LSTM', 'Transformer']
        
        for i, model_name in enumerate(models):
            # Model container
            model_frame = Frame(results_frame, bg="white", relief="raised", bd=2)
            model_frame.pack(fill=BOTH, expand=True, pady=5)
            
            # Model name
            name_label = Label(
                model_frame,
                text=f"🤖 {model_name}",
                font=("Arial", 12, "bold"),
                bg="white",
                fg="#1976d2"
            )
            name_label.pack(anchor="w", padx=10, pady=(5, 0))
            
            # Sentiment result
            sentiment_label = Label(
                model_frame,
                text="Waiting for input...",
                font=("Arial", 11),
                bg="white"
            )
            sentiment_label.pack(anchor="w", padx=15, pady=2)
            
            # Probabilities
            prob_label = Label(
                model_frame,
                text="",
                font=("Arial", 9),
                bg="white",
                fg="#666",
                justify="left"
            )
            prob_label.pack(anchor="w", padx=15, pady=(0, 5))
            
            # Confidence bar
            confidence_frame = Frame(model_frame, bg="white")
            confidence_frame.pack(fill=X, padx=15, pady=(0, 5))
            
            confidence_bar = Canvas(
                confidence_frame,
                height=20,
                bg="#e0e0e0",
                highlightthickness=0
            )
            confidence_bar.pack(fill=X)
            
            confidence_label = Label(
                confidence_frame,
                text="",
                font=("Arial", 9),
                bg="white"
            )
            confidence_label.pack(anchor="e", pady=2)
            
            self.result_frames[model_name] = {
                'sentiment': sentiment_label,
                'probs': prob_label,
                'bar': confidence_bar,
                'confidence_label': confidence_label
            }
    
    def predict(self):
        """Make predictions with all 3 models"""
        text = self.text_widget.get_text().strip()
        
        if not text:
            messagebox.showwarning("Input Required", "Please enter some text to analyze")
            return
        
        try:
            # Get predictions
            results = predict_sentiment(text)
            
            # Display results for each model
            colors = {'Positive': '#4CAF50', 'Neutral': '#FF9800', 'Negative': '#f44336'}
            
            for model_name, result in results.items():
                sentiment = result['sentiment']
                confidence = result['confidence']
                probs = result['probabilities']
                
                # Update sentiment label
                sentiment_text = f"Sentiment: {sentiment} ({confidence:.1%})"
                self.result_frames[model_name]['sentiment'].config(
                    text=sentiment_text,
                    fg=colors.get(sentiment, '#000')
                )
                
                # Update probabilities
                prob_text = (
                    f"Negative: {probs['Negative']:.1%}  |  "
                    f"Neutral: {probs['Neutral']:.1%}  |  "
                    f"Positive: {probs['Positive']:.1%}"
                )
                self.result_frames[model_name]['probs'].config(text=prob_text)
                
                # Update confidence bar
                bar = self.result_frames[model_name]['bar']
                bar.delete("all")
                
                # Background
                bar.create_rectangle(0, 0, 300, 20, fill="#e0e0e0", outline="")
                
                # Confidence bar
                bar_width = confidence * 300
                bar.create_rectangle(0, 0, bar_width, 20, fill=colors.get(sentiment, '#2196F3'), outline="")
                
                # Text on bar
                bar.create_text(150, 10, text=f"{confidence:.1%}", fill="white", font=("Arial", 9, "bold"))
                
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def clear(self):
        """Clear text and results"""
        self.text_widget.clear()
        
        for model_name in self.result_frames:
            self.result_frames[model_name]['sentiment'].config(text="Waiting for input...")
            self.result_frames[model_name]['probs'].config(text="")
            bar = self.result_frames[model_name]['bar']
            bar.delete("all")


if __name__ == "__main__":
    root = Tk()
    app = SentimentApp(root)
    root.mainloop()
