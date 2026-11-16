import sys, os, time
import numpy as np
import torch, torch.nn as nn

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N_PER = int(sys.argv[2]) if len(sys.argv) > 2 else 93      # training scans taken from EACH sequence
TAG = f"multi{N_PER}_"
torch.manual_seed(SEED); np.random.seed(SEED)
rng = np.random.default_rng(SEED)

ROAD_IDS = [40, 44]
X_RANGE, Y_RANGE = 40, 20
N_POINTS = 4096
SCALE = np.array([X_RANGE, Y_RANGE, 3.0], dtype=np.float32)
EPOCHS, BATCH = 30, 16

def load_crop(seq, idx):
    d = f"data/dataset/sequences/{seq}"
    pts = np.fromfile(f"{d}/velodyne/{idx:06d}.bin", dtype=np.float32).reshape(-1, 4)[:, :3]
    lab = np.fromfile(f"{d}/labels/{idx:06d}.label", dtype=np.uint32) & 0xFFFF
    keep = (np.abs(pts[:, 0]) < X_RANGE) & (np.abs(pts[:, 1]) < Y_RANGE)
    return pts[keep], np.isin(lab, ROAD_IDS)[keep]

train_ids = [("04", i) for i in range(N_PER)] + [("00", i) for i in range(N_PER)]
tests = {"04": [("04", i) for i in range(200, 271)],      # same drive as training, later scans
         "07": [("07", i) for i in range(0, 300)]}        # a different drive, never seen
print(f"train scans: {N_PER} from sequence 04 + {N_PER} from sequence 00")
print("loading scans...")
data = {k: load_crop(*k) for k in train_ids}
test_data = {name: [load_crop(*k) for k in ids] for name, ids in tests.items()}

class PointNetSeg(nn.Module):
    def __init__(self, k=2):
        super().__init__()
        self.f1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                                nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.f2 = nn.Sequential(nn.Conv1d(128, 512, 1), nn.BatchNorm1d(512), nn.ReLU())
        self.head = nn.Sequential(nn.Conv1d(128 + 512, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
                                  nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                                  nn.Conv1d(128, k, 1))
    def forward(self, x):
        f = self.f1(x)
        g = self.f2(f).max(2, keepdim=True)[0].expand(-1, -1, x.shape[2])
        return self.head(torch.cat([f, g], 1))

def sample_batch(keys):
    X, Y = [], []
    for k in keys:
        pts, gt = data[k]
        sel = rng.choice(len(pts), N_POINTS, replace=len(pts) < N_POINTS)
        p = pts[sel] / SCALE
        if rng.random() < 0.5:
            p[:, 1] *= -1                      # mirror left/right
        X.append(p.T); Y.append(gt[sel])
    return (torch.tensor(np.stack(X), dtype=torch.float32),
            torch.tensor(np.stack(Y).astype(np.int64)))

model = PointNetSeg()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

print("training...")
t0 = time.time()
for epoch in range(EPOCHS):
    model.train()
    order = rng.permutation(len(train_ids))
    losses = []
    for b in range(0, len(order), BATCH):
        keys = [train_ids[j] for j in order[b:b + BATCH]]
        X, Y = sample_batch(keys)
        opt.zero_grad()
        loss = loss_fn(model(X), Y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"epoch {epoch + 1}/{EPOCHS}  loss {np.mean(losses):.4f}  ({time.time() - t0:.0f}s)")

os.makedirs("results", exist_ok=True)
torch.save(model.state_dict(), f"results/pointnet_{TAG}seed{SEED}.pt")

@torch.no_grad()
def predict_full(pts):
    model.eval()
    n = len(pts)
    perm = rng.permutation(n)
    pad = (-n) % N_POINTS
    if pad:
        perm = np.concatenate([perm, rng.choice(n, pad, replace=True)])
    pred = np.zeros(n, dtype=bool)
    p = pts / SCALE
    for c in range(0, len(perm), N_POINTS):
        idx = perm[c:c + N_POINTS]
        x = torch.tensor(p[idx].T[None], dtype=torch.float32)
        pred[idx] = model(x).argmax(1)[0].numpy().astype(bool)
    return pred

def iou_f1(pred, gt):
    tp, fp, fn = np.sum(pred & gt), np.sum(pred & ~gt), np.sum(~pred & gt)
    return tp / (tp + fp + fn + 1e-9), 2 * tp / (2 * tp + fp + fn + 1e-9)

out = {}
for name, scans in test_data.items():
    res = np.array([iou_f1(predict_full(pts), gt) for pts, gt in scans])
    out[name] = (res[:, 0].mean(), res[:, 1].mean())
    print(f"test on sequence {name}: mean IoU {out[name][0]:.3f} | mean F1 {out[name][1]:.3f}")
print(f"RESULT_MULTI seed={SEED} n_per_seq={N_PER} "
      f"seq04_iou={out['04'][0]:.4f} seq04_f1={out['04'][1]:.4f} "
      f"seq07_iou={out['07'][0]:.4f} seq07_f1={out['07'][1]:.4f}")