def parse(file):
    (str_height, str_width, str_names) = file.readlines()[0].split("*")
    (height, width) = (int(str_height), int(str_width))

    names = str_names.strip("][").replace("'", "").split(", ")
    return (height,width, names)

if __name__ == "__main__":
    file = open("data_text", "r")
    parse(file)