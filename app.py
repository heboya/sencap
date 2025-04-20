from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
from torchvision import transforms
import io
import base64
from torch import nn

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS so frontend can talk to backend

class CVAE(nn.Module):
    def __init__(self, latent_size):
        super(CVAE, self).__init__()
        self.class_size = 31  # number of features
        self.feature_size = 96 * 96 * 1  # image size

        # Convolutional encoder for the image
        self.c1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1) # -> 48x48x32
        self.c2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # -> 24x24x64
        self.fc_img = nn.Linear(24 * 24 * 64, 400)

        # Fully connected layers for latent space
        self.fc21 = nn.Linear(400 + self.class_size, latent_size)
        self.fc22 = nn.Linear(400 + self.class_size, latent_size)

        # Decoder (Fully connected and deconvolutional layers)
        self.fc_dec = nn.Linear(latent_size + self.class_size, 24 * 24 * 64)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def encode(self, x, c):  # Q(z|x, c)
        # Convolutions on the image
        h1 = self.elu(self.c1(x))
        h1 = self.elu(self.c2(h1))
        h1 = h1.view(h1.size(0), -1)  # Flatten for the fully connected layer

        # Fully connected layer for image encoding
        img_features = self.elu(self.fc_img(h1))

        # Concatenate image features with additional features
        combined = torch.cat([img_features, c], dim=1)

        # Latent space projections
        z_mu = self.fc21(combined)
        z_var = self.fc22(combined)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):  # P(x|z, c)
        # Concatenate latent vector with class conditioning
        x = torch.cat((z, c), dim=1)  # (batch_size, latent_size + class_size)

        # Fully connected layer and reshape
        x = self.relu(self.fc_dec(x))
        x = x.view(-1, 64, 24, 24)

        # Deconvolutional layers
        x = self.relu(self.deconv1(x))
        x = self.sigmoid(self.deconv2(x))
        return x

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

# Load your ML model (change the path to your model file)
model = torch.load("./TEST.pth", weights_only=False, map_location="cpu") 
device = torch.device("cpu")
model.to(device)  # Ensure it's on the correct device  
model.eval()  # Set to evaluation mode

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data.get("features")

        if not features or len(features) != 31:
            return jsonify({"error": "Input vector must have 31 normalized features"}), 400

        # Convert input to tensor
        X = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

        generated_samples = []

        with torch.no_grad():
            for _ in range(16):
                z = torch.randn(1, 1).to(device)  # Adjust latent dim if needed
                sample = model.decode(z, X).cpu()  # shape: (1, 1, 96, 96)
                generated_samples.append(sample)

        to_pil = transforms.ToPILImage()
        base64_images = []

        for sample in generated_samples:
            img_tensor = sample.squeeze(0)  # shape: (1, 96, 96)
            img = to_pil(img_tensor)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            base64_images.append(img_str)

        return jsonify({"images": base64_images})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



# Start the server
if __name__ == "__main__":
    app.run(debug=True)
