import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from glob import glob
from tqdm import tqdm  # ✅ 加在顶部
# ---------- Dataset ----------
class StateActionSequenceDataset(Dataset):
    def __init__(self, folder, seq_len=100):
        self.seq_len = seq_len
        self.samples = []

        files = glob(os.path.join(folder, "*.npz"))
        for file in files:
            data = np.load(file,allow_pickle=True)
            state = data["norm_state"]
            action = data["norm_action"]

            if state.shape[0] <= seq_len:
                continue
            sa = np.concatenate([state, action], axis=1)  # (T, D_s + D_a)

            for i in range(state.shape[0] - seq_len):
                input_seq = sa[i:i+seq_len]
                target_seq = state[i+1:i+seq_len+1]
                self.samples.append((input_seq.astype(np.float32), target_seq.astype(np.float32)))

    def __len__(self):
        return len(self.samples)

    # def __getitem__(self, idx):
    #
    #     x, y = self.samples[idx]
    #     # 在 __getitem__ 里加一行 debug
    #     # if np.isnan(x).any() or np.isnan(y).any():
    #     #     print("Found NaN in sample:", idx)
    #
    #     return torch.tensor(x), torch.tensor(y)
    def __getitem__(self, idx):
        for attempt in range(10):
            x, y = self.samples[idx]
            if not (np.isnan(x).any() or np.isnan(y).any()):
                return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

            # ❗有 NaN，就重新随机抽一个
            idx = np.random.randint(0, len(self.samples))

        raise ValueError("Exceeded max retries due to too many NaNs in dataset.")
# ---------- Model ----------
class LSTMStatePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)  # x: (B, T, D)
        out = self.linear(out)  # (B, T, state_dim)
        return out

# ---------- Config ----------

if __name__ == '__main__':

    SEQ_LEN = 100
    BATCH_SIZE = 64
    EPOCHS = 100
    LR = 1e-3
    HIDDEN_DIM = 128
    FOLDER = "normalized_npz"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)
    MODEL_SAVE_PATH = "lstm_model.pt"

    # ---------- Data ----------
    dataset = StateActionSequenceDataset(FOLDER, seq_len=SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    input_dim = dataset[0][0].shape[-1]
    output_dim = dataset[0][1].shape[-1]

    # ---------- Model Setup ----------
    model = LSTMStatePredictor(input_dim, HIDDEN_DIM, output_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # ---------- Training ----------
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)
        # for xb, yb in dataloader:
        for xb, yb in loop:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.6f}")

    # ---------- Save Model ----------
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"\n✅ Model saved to '{MODEL_SAVE_PATH}'")
