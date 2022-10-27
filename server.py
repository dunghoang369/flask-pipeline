import os
import io
from waitress import serve
import flask
from flask import Flask, Response, make_response, request, jsonify
from flask_cors import CORS
from waitress import serve
import torch
from pytorch_images_classification import CNN
import torchvision.transforms as transforms
from PIL import Image

app = Flask(__name__)
CORS(app)

model = torch.load('./model.pth')
model.eval()

print(model)

@app.route('/health', methods=['GET', 'POST']) # URI
def health_check():
    return Response(status=200)



def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))

    return my_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return predicted_idx

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        print(request.files['file'])
        file = request.files['file']
        img_bytes = file.read()
        predicted_idx= get_prediction(image_bytes=img_bytes)
        return jsonify({'predicted_idx': predicted_idx})

if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 5003))
    except:
        port = 5003

    print("Starting server on port {}".format(port))
    serve(app, host='0.0.0.0', port=port)
