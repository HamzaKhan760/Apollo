import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
from PIL import Image
import random
from backend.deepfake_model import DeepfakeCNNLSTM

def frame_number(frame_path):
    return int(frame_path.rsplit('_frame_', 1)[1].removesuffix('.jpg'))

class DeepfakeFrameDataset(Dataset):
    def __init__(self, frame_root, sequence_length=10, transform=None):
        self.sequence_length = sequence_length
        self.transform = transform
        self.samples = []

        for label, subfolder in enumerate(['real', 'fake']):
            subfolder_path = os.path.join(frame_root, subfolder)
            video_frames = {}

            for filename in os.listdir(subfolder_path):
                if filename.endswith('.jpg'):
                    video_id = filename.rsplit('_frame_', 1)[0]
                    video_frames.setdefault(video_id, []).append(os.path.join(subfolder_path, filename))

            for video_id, frame_paths in video_frames.items():
                frame_paths.sort(key=frame_number)
                if len(frame_paths) >= sequence_length:
                    self.samples.append((frame_paths, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        max_start = len(frame_paths) - self.sequence_length
        start_idx = random.randint(0, max_start)
        selected_frames = frame_paths[start_idx: start_idx + self.sequence_length]

        frames = []
        for frame_path in selected_frames:
            img = Image.open(frame_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        frames = torch.stack(frames)
        return frames, torch.tensor(label, dtype=torch.float32)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = DeepfakeFrameDataset(frame_root='extracted_frames', sequence_length=10, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    model = DeepfakeCNNLSTM().to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for frames, labels in train_loader:
            frames, labels = frames.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "deepfake_cnn_lstm.pth")
