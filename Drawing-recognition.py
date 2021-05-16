from tkinter import *
from tkinter.colorchooser import askcolor
from PIL import ImageGrab
import numpy as np
import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.tensor_shape import as_dimension
from parse import parse

class Board(object):

    default_brush_size = 5.0

    def __init__(self):
        self.root = Tk()
        self.root.geometry("1680x980")

        self.brush_button = Button(self.root, text="Brush", command=self.use_brush)
        self.brush_button.grid(row=0, column=1)

        self.eraser_button = Button(self.root, text="Eraser", command=self.use_eraser)
        self.eraser_button.grid(row=0, column=2)

        self.guess_button = Button(self.root, text="Guess", command=self.save_guess)
        self.guess_button.grid(row=0, column=3)

        self.color_display = Button(self.root, bg = "black", height=2, width=6)
        self.color_display.grid(row=1, column=0)

        self.color_button = Button(self.root, text="Color", command=self.choose_color)
        self.color_button.grid(row=1, column=1)

        self.choose_size_button = Scale(self.root, from_=1, to=10, orient=HORIZONTAL)
        self.choose_size_button.grid(row=1, column=2)

        self.clear_button = Button(self.root, text="Clear", command=self.clear)
        self.clear_button.grid(row=1, column=3)

        self.canvas = Canvas(self.root, bg="white", width=1050, height=800)
        self.canvas.grid(row=2, columnspan=4)

        self.count = 0 

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = "black"
        self.eraser_on = False
        self.active_button = self.brush_button
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

    def use_brush(self):
        self.activate_button(self.brush_button)

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def choose_color(self):
        self.eraser_on = False
        newcolor = askcolor(color = self.color)[1]
        if newcolor != None:
            self.color = newcolor
            self.color_display["bg"] = newcolor
        else:
            return None
    
    def clear(self):
        self.canvas.delete("all")

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def draw(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = "white" if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y, width=self.line_width, fill=paint_color, capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y
    
    def save_guess(self):
        #save
        self.canvas.update()
        x=self.root.winfo_rootx()+self.canvas.winfo_x()
        y=self.root.winfo_rooty()+self.canvas.winfo_y()
        x1=x+self.canvas.winfo_width()
        y1=y+self.canvas.winfo_height()
        ImageGrab.grab().crop((x,y,x1,y1)).save(f"temp{self.count}.png")

        #guess
        new_path = pathlib.Path(__file__).with_name(f"temp{self.count}.png")
        img = keras.preprocessing.image.load_img(new_path, target_size=(img_height, img_width))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch
        
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        # print(score)
        results = tf.math.top_k(score, k=3)
        # print(results.values.numpy())
        # print(results.indices.numpy())

        top3count = 0
        for i in results.indices.numpy():
            print(
                "This image most likely belongs to {} with a {:.2f} percent confidence."
                .format(class_names[i], 100 * results.values.numpy()[top3count])
            )
            top3count += 1

        self.count +=1

    def reset(self, event):
        self.old_x, self.old_y = None, None

if __name__ == "__main__":
    model = keras.models.load_model("trained model\saved_model")
    file = open("data_text", "r")
    (img_height, img_width, class_names) = parse(file)
    Board()