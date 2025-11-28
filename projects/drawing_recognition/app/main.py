import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF messages
from tkinter import *
from PIL import ImageGrab, Image
from canvas_widget import DrawCanvas
from predictor import predict
from io import BytesIO

# Initialize Tkinter
root = Tk()
root.title("Drawing Recognition")

# Canvas
canvas = DrawCanvas(root)

# label to show prediction
result_label = Label(root, text= "draw something ..." , font=("Arial", 20))
result_label.pack(pady=10)

# predict when button is clicked
def get_prediction():
    # Create image from canvas using PostScript
    ps = canvas.postscript(colormode='gray')
    
    # Convert PostScript to PIL Image
    from io import BytesIO
    img = Image.open(BytesIO(ps.encode('utf-8')))
    
    # Resize to 28x28 and convert to grayscale
    img = img.convert('L').resize((28, 28), Image.Resampling.LANCZOS)
    
    # Invert colors (canvas is white bg with black drawing, model expects opposite)
    img = Image.eval(img, lambda x: 255 - x)
    
    label, prob = predict(img)
    result_label.config(text=f"{label} ({prob:.2f})")

# clear button 
def clear():
    canvas.delete("all")
    result_label.config(text="draw something...")

# Buttons frame
button_frame = Frame(root)
button_frame.pack(pady=10)

Button(button_frame, text="Predict", command=get_prediction, bg="green", fg="white", width=10).pack(side=LEFT, padx=5)
Button(button_frame, text="Clear", command=clear, bg="red", fg="white", width=10).pack(side=LEFT, padx=5)

# Run the application
root.mainloop()