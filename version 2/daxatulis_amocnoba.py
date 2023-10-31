import tkinter as tk
from PIL import Image, ImageDraw
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input


class PaintApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Paint App")
        self.pen_color = "black"
        self.image_size = (28, 28) # რამდენი რამდენზეა პიქსელების რაოდენობა
        self.pixel_size = 30 
        self.create_widgets()

        # სუფთა canva-ს შექმნა
        self.image = Image.new('RGB', self.image_size, (255, 255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)

    def create_widgets(self):
        self.canvas = tk.Canvas(self.master, bg="white", width=self.image_size[0]*self.pixel_size, height=self.image_size[1]*self.pixel_size)
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)

        clear_button = tk.Button(self.master, text="Clear", command=self.clear_canvas)
        clear_button.pack(side=tk.LEFT, padx=10)

        save_button = tk.Button(self.master, text="Save", command=self.save_as_png)
        save_button.pack(side=tk.LEFT)

        quit_button = tk.Button(self.master, text="Quit", command=self.master.destroy)
        quit_button.pack(side=tk.LEFT, padx=10)

    def draw_on_canvas(self, event):
        x, y = event.x // self.pixel_size, event.y // self.pixel_size
        if 0 <= x < self.image_size[0] and 0 <= y < self.image_size[1]:
            # draw the pixel and mark it as colored
            self.canvas.create_rectangle(x*self.pixel_size, y*self.pixel_size,
                                          (x+1)*self.pixel_size, (y+1)*self.pixel_size,
                                          fill=self.pen_color, outline="")

            self.draw.rectangle([x, y, x+1, y+1], fill=self.pen_color)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new('RGB', self.image_size, (255, 255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)

    def save_as_png(self):
        file_name = "1.png"
        script_dir = os.path.dirname(os.path.abspath(__file__)) # შენახვის ადგილი
        file_path = os.path.join(script_dir, file_name)   # საბოლოო შენახვის ადგილი
        self.image.save(file_path)   #fotos Senaxva
        # print(f"sheinaxa rogorc {file_name}.")  # შენახული ფოტოს სახელის ნახვა თუ მოგინდება
        self.master.destroy()



if __name__ == "__main__":
    root = tk.Tk()
    app = PaintApp(root)
    root.mainloop()




image_path = "1.png"
image = Image.open(image_path).convert('L')  #grayscale-ში გადაჰყავს ფოტო
image = image.resize((28, 28))  # ფოტოს სწორ ზომაზე გადაყვანა
image_array = np.array(image)  # numpy array ში გადაყვანა
image_array = image_array / 255.0
image_array = 1 - image_array  # Invert the image (if your training data is white digits on a black background)
data = image_array.reshape(1, 28, 28, 1)  # Reshape to match the input shape of your model



loaded_model = load_model("mnist_neural1.h5")
# loaded_model = joblib.load('mnist_neural.h5')    #შენახული მოდელის ჩატვირთვა
# predicted_label = loaded_model.predict(data)
predicted_label = loaded_model.predict(data)

# predicted_digit = np.argmax(predicted_label)
predicted_digit = np.argmax(predicted_label)


highest_probability = predicted_label[0][predicted_digit]


# print("შეიძლება ეს იყოს:", predicted_label)   # გამოაქვს რა იფიქრა მოდელმა. (ფოტოზე რა ეხატა მისი აზრით)

print("გამოცნობილი:", predicted_digit)
print("ალბათობა:", highest_probability)




