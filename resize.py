from PIL import Image
import pathlib 

def resize(image):
    """resize the image inputted, so it matches the image size of our paint application"""
    newimg = image.resize((1050,800))
    return newimg

p = pathlib.Path('.//training/')

# All subdirectories in the current directory, not recursive.
for f in p.iterdir():
    for train_im in f.iterdir():
        with Image.open(train_im) as inimg:
            newimg = resize(inimg)
            im = newimg.save(train_im.resolve(), "PNG")
print("update success")