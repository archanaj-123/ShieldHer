"""
lstm_classifier.py
Trained classes:
  0 = Normal       (standing / sitting / walking)
  1 = Fighting     (Hockey Fight + Real Life Violence)
  2 = Person Down  (Le2i Fall + Lie + Likefall)
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, deque

CLASS_NAMES  = ["Normal", "Fighting", "Person Down"]
SEQ_LEN      = 30
FRAME_W      = 640
FRAME_H      = 480
CONF_MIN     = 0.65   # raw confidence needed to count as a vote
VOTE_WINDOW  = 12     # look at last N predictions
VOTE_NEEDED  = 8      # need this many matching votes to fire alert

CLASS_RISK = {"Normal": 0, "Fighting": 9, "Person Down": 8}


class PoseLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(34, 128, num_layers=2,
                            batch_first=True, dropout=0.4, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, 3)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class LSTMClassifier:
    def __init__(self, model_path="pose_lstm.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = PoseLSTM().to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()
        self.kp_buf   = defaultdict(lambda: deque(maxlen=SEQ_LEN))
        self.vote_buf = defaultdict(lambda: deque(maxlen=VOTE_WINDOW))
        print(f"[LSTM] Loaded on {self.device}")

    def update(self, pid, kps_17x2):
        kps = np.array(kps_17x2, dtype=np.float32).copy()
        kps[:, 0] = np.clip(kps[:, 0] / FRAME_W, 0, 1)
        kps[:, 1] = np.clip(kps[:, 1] / FRAME_H, 0, 1)
        self.kp_buf[pid].append(kps.flatten())

    def raw_predict(self, pid):
        if len(self.kp_buf[pid]) < SEQ_LEN:
            return "Normal", 0.0
        seq = np.array(self.kp_buf[pid], dtype=np.float32)
        x   = torch.tensor(seq).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]
        idx = int(probs.argmax())
        return CLASS_NAMES[idx], float(probs[idx])

    def predict(self, pid):
        label, conf = self.raw_predict(pid)
        self.vote_buf[pid].append(label if conf >= CONF_MIN else "Normal")
        votes = list(self.vote_buf[pid])
        if len(votes) < VOTE_WINDOW:
            return "Normal", 0.0
        for cls in ["Fighting", "Person Down"]:
            n = votes.count(cls)
            if n >= VOTE_NEEDED:
                return cls, round(n / VOTE_WINDOW, 2)
        return "Normal", 0.0

    def get_risk(self, pid):
        label, conf = self.predict(pid)
        return CLASS_RISK.get(label, 0), label, conf

    def buf_size(self, pid):
        return len(self.kp_buf[pid])

    def all_probs(self, pid):
        if len(self.kp_buf[pid]) < SEQ_LEN:
            return {c: 0.0 for c in CLASS_NAMES}
        seq = np.array(self.kp_buf[pid], dtype=np.float32)
        x   = torch.tensor(seq).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0].cpu().numpy()
        return {CLASS_NAMES[i]: float(probs[i]) for i in range(3)}
