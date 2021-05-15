from tkinter import *
from tkinter.colorchooser import askcolor
from PIL import Image
import numpy as np
import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.tensor_shape import as_dimension


class Board(object):

    default_brush_size = 5.0

    def __init__(self):
        self.root = Tk()
        self.root.geometry('1680x980')

        self.brush_button = Button(self.root, text='Brush', command=self.use_brush)
        self.brush_button.grid(row=0, column=1)

        self.eraser_button = Button(self.root, text='Eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=2)

        self.guess_button = Button(self.root, text='Guess', command=self.guess)
        self.guess_button.grid(row=0, column=3)

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
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y, width=self.line_width, fill=paint_color, capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y
    
    # def save(self):
    #     self.canvas.postscript(file="live.eps")
    #     saved = Image.open("live.eps")
    #     saved.save("live.png", "png")
    #     with Image.open("live.png") as im:
    #         im.show()

    def guess(self):

        new_path = pathlib.Path(__file__).with_name("unknown2.png")

        img = keras.preprocessing.image.load_img(new_path, target_size=(img_height, img_width))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch
        
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        results = tf.math.top_k(score, k=3)
        print(results.values.numpy())
        print(results.indices.numpy())

        # for i in results.indices.numpy():
        #     print(
        #         "This image most likely belongs to {} with a {:.2f} percent confidence."
        #         .format(class_names[i], 100 * results.values.numpy())
        #     )

    def reset(self, event):
        self.old_x, self.old_y = None, None

if __name__ == "__main__":
    
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential

    #directory
    data_dir = pathlib.Path("training")
    print(data_dir)

    #total images
    image_count = len(list(data_dir.glob('*/*.png')))
    print(image_count)

    #show first image
    # ones = list(data_dir.glob('1/*'))
    # pic = PIL.Image.open(str(ones[0]))
    # pic.show()

    #separate training set and validation set
    batch_size = 32
    img_height = 412
    img_width = 504

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=102,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    #print all class names
    class_names = train_ds.class_names
    # print(class_names)


    # for image_batch, labels_batch in train_ds:
    #     print(image_batch.shape)
    #     print(labels_batch.shape)
    #     break

    #configure dataset
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    #standardize colors
    normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixels values are now in `[0,1]`.
    # print(np.min(first_image), np.max(first_image))


    #number of classes
    num_classes = 10

    #augmentation
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )

    #model creation with augmentation and dropout
    model = Sequential([
        data_augmentation,
        layers.experimental.preprocessing.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    #model compilation
    model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    #training
    epochs=10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    Board()