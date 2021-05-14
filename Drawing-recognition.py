from tkinter import *
from tkinter.colorchooser import askcolor

class Board(object):

    default_brush_size = 5.0

    def __init__(self):
        self.root = Tk()
        self.root.geometry('1680x980')

        self.brush_button = Button(self.root, text='Brush', command=self.use_brush)
        self.brush_button.grid(row=0, column=1)

        self.eraser_button = Button(self.root, text='Eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=2)

        self.color_display = Button(self.root, bg = "black", height=2, width=6)
        self.color_display.grid(row=1, column=0)

        self.color_button = Button(self.root, text='Color', command=self.choose_color)
        self.color_button.grid(row=1, column=1)

        self.choose_size_button = Scale(self.root, from_=1, to=10, orient=HORIZONTAL)
        self.choose_size_button.grid(row=1, column=2)

        self.clear_button = Button(self.root, text='Clear', command=self.clear)
        self.clear_button.grid(row=1, column=3)

        self.canvas = Canvas(self.root, bg='white', width=1050, height=800)
        self.canvas.grid(row=2, columnspan=4)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = "black"
        self.eraser_on = False
        self.active_button = self.brush_button
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.reset)

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
        self.canvas.delete('all')

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def draw(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

if __name__ == '__main__':
    Board()