from tkinter import Canvas

class DrawCanvas(Canvas):
    def __init__(self, parent):
        super().__init__(parent, width=280, height=280, bg='white')
        self.pack()
        self.bind('<B1-Motion>', self.paint)
    def paint(self , event):
        x , y = event.x , event.y
        r = 8
        self.create_oval(x-r, y-r, x+r, y+r, fill='black')