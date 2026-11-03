import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import os
from PIL import Image
import shutil

def predict_deepfake(video_path, model, transform, device, sequence_length=10):
    frame_output_dir = "temp_frames"
    os.makedirs(frame_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(frame_output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()

    if frame_count < sequence_length:
        shutil.rmtree(frame_output_dir)
        raise ValueError(f"Video must have at least {sequence_length} frames.")

    frame_paths = sorted([os.path.join(frame_output_dir, f) for f in os.listdir(frame_output_dir) if f.endswith('.jpg')])
    selected_frames = frame_paths[:sequence_length]

    frames = []
    for frame_path in selected_frames:
        img = Image.open(frame_path).convert("RGB")
        img = transform(img)
        frames.append(img)

    frames_tensor = torch.stack(frames).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(frames_tensor)
        prob = torch.sigmoid(outputs).item()

    shutil.rmtree(frame_output_dir)

    prediction = "FAKE" if prob >= 0.5 else "REAL"
    return prob, prediction

class DeepfakeCNNLSTM(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=1):
        super(DeepfakeCNNLSTM, self).__init__()
        cnn = models.resnet18(pretrained=True)
        cnn.fc = nn.Identity()
        self.cnn = cnn
        self.feature_dim = 512
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, frames):
        batch_size, seq_len, C, H, W = frames.shape
        features = []
        for i in range(seq_len):
            frame = frames[:, i, :, :, :]
            f = self.cnn(frame)
            features.append(f)
        features = torch.stack(features, dim=1)
        lstm_out, _ = self.lstm(features)
        last_output = lstm_out[:, -1, :]
        logits = self.fc(last_output)
        return logits.squeeze(1)