from tkinter import Frame, Text, Scrollbar, RIGHT, Y, BOTH, LEFT


class TextInputWidget(Frame):
    """
    A text input widget with scrollbar
    Similar to DrawCanvas but for text input
    """
    def __init__(self, parent, height=8, width=50):
        super().__init__(parent)
        
        # Create scrollbar
        scrollbar = Scrollbar(self)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        # Create text widget
        self.text = Text(
            self,
            height=height,
            width=width,
            yscrollcommand=scrollbar.set,
            font=("Arial", 12),
            wrap="word",
            bg="#f0f0f0"
        )
        self.text.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.config(command=self.text.yview)
    
    def get_text(self):
        """Get the text content"""
        return self.text.get("1.0", "end-1c")
    
    def clear(self):
        """Clear the text widget"""
        self.text.delete("1.0", "end")
    
    def set_text(self, text):
        """Set the text content"""
        self.clear()
        self.text.insert("1.0", text)
