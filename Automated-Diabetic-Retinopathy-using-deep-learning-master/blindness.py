import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from torch import nn
import torchvision
from torchvision import models
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your model architecture
model = models.resnet152(pretrained=False)
num_ftrs = model.fc.in_features
out_ftrs = 5
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Linear(512, out_ftrs),
    nn.LogSoftmax(dim=1)
)

# Load model checkpoint
def load_model(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)  # Set weights_only=False
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

# Inference function
def inference(model, file, transform, classes):
    file = Image.open(file).convert('RGB')
    img = transform(file).unsqueeze(0)
    img = img.to(device)

    model.eval()
    with torch.no_grad():
        out = model(img)
        ps = torch.exp(out)
        top_p, top_class = ps.topk(1, dim=1)
        value = top_class.item()
        return value, classes[value]

# Image transforms for inference
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# Define class names
classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# Create the Tkinter GUI
class BlindnessDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Retinal Blindness Detection")
        self.root.geometry("600x500")

        # Button to upload an image
        self.upload_button = tk.Button(self.root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=20)

        # Label to display image
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)

        # Label to display prediction result
        self.result_label = tk.Label(self.root, text="Prediction Result: ", font=("Arial", 14))
        self.result_label.pack(pady=20)

        # Load the model
        self.model = load_model('classifier.pt')  # Make sure the model path is correct

    def upload_image(self):
        # Open file dialog to choose an image
        file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])

        if file_path:
            self.display_image(file_path)
            self.make_prediction(file_path)

    def display_image(self, image_path):
        # Open the image and display it on the GUI
        img = Image.open(image_path)
        img = img.resize((250, 250))  # Resize the image to fit in the window
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk  # Keep a reference to the image

    def make_prediction(self, image_path):
        # Perform inference and display the result
        try:
            value, prediction = inference(self.model, image_path, test_transforms, classes)
            self.result_label.config(text=f"Prediction Result: {prediction} (Severity Level: {value})")
        except Exception as e:
            messagebox.showerror("Error", f"Error in prediction: {e}")

# Run the application
if __name__ == '__main__':
    root = tk.Tk()
    app = BlindnessDetectionApp(root)
    root.mainloop()
