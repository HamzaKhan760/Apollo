from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from deepfake_model import DeepfakeCNNLSTM, predict_deepfake
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DeepfakeCNNLSTM().to(device)
model.load_state_dict(torch.load('deepfake_cnn_lstm.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400

    video_file = request.files['video']
    video_path = os.path.join("temp_upload.mp4")
    video_file.save(video_path)

    try:
        prob, prediction = predict_deepfake(video_path, model, transform, device)
        os.remove(video_path)
        return jsonify({'probability': prob, 'prediction': prediction})
    except Exception as e:
        os.remove(video_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
