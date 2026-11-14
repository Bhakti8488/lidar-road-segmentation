import os, time
import numpy as np
import torch, torch.nn as nn
import matplotlib.pyplot as plt
import open3d as o3d

torch.manual_seed(0); np.random.seed(0); o3d.utility.random.seed(0)

SEQ_DIR = "data/dataset/sequences/04"
ROAD_IDS = [40, 44]
X_RANGE, Y_RANGE = 40, 20
N_POINTS = 4096
SCALE = np.array([X_RANGE, Y_RANGE, 3.0], dtype=np.float32)   # squeeze coordinates to about -1..1
TRAIN_IDS = list(range(0, 186))
TEST_IDS = list(range(200, 271))
EPOCHS, BATCH = 30, 16

NAMES = {0: "unlabeled", 1: "outlier", 10: "car", 11: "bicycle", 13: "bus", 15: "motorcycle",
         16: "on-rails", 18: "truck", 20: "other-vehicle", 30: "person", 31: "bicyclist",
         32: "motorcyclist", 40: "road", 44: "parking", 48: "sidewalk", 49: "other-ground",
         50: "building", 51: "fence", 52: "other-structure", 60: "lane-marking",
         70: "vegetation", 71: "trunk", 72: "terrain", 80: "pole", 81: "traffic-sign",
         99: "other-object"}

# ---------- data ----------
def load_crop(idx):
    pts = np.fromfile(f"{SEQ_DIR}/velodyne/{idx:06d}.bin", dtype=np.float32).reshape(-1, 4)[:, :3]
    lab = np.fromfile(f"{SEQ_DIR}/labels/{idx:06d}.label", dtype=np.uint32) & 0xFFFF
    keep = (np.abs(pts[:, 0]) < X_RANGE) & (np.abs(pts[:, 1]) < Y_RANGE)
    return pts[keep], np.isin(lab, ROAD_IDS)[keep], lab[keep]

print("loading scans...")
data = {i: load_crop(i) for i in TRAIN_IDS + TEST_IDS}

# ---------- baseline (same as before, on the cropped points) ----------
def ransac_road(pts):
    cand = np.where(pts[:, 2] < -1.0)[0]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[cand]))
    plane, _ = pcd.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=1000)
    normal = np.array(plane[:3])
    dist = np.abs(pts @ normal + plane[3]) / np.linalg.norm(normal)
    return dist < 0.2

def iou_f1(pred, gt):
    tp, fp, fn = np.sum(pred & gt), np.sum(pred & ~gt), np.sum(~pred & gt)
    return tp / (tp + fp + fn + 1e-9), 2 * tp / (2 * tp + fp + fn + 1e-9)

# ---------- PointNet ----------
class PointNetSeg(nn.Module):
    def __init__(self, k=2):
        super().__init__()
        self.f1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                                nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.f2 = nn.Sequential(nn.Conv1d(128, 512, 1), nn.BatchNorm1d(512), nn.ReLU())
        self.head = nn.Sequential(nn.Conv1d(128 + 512, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
                                  nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                                  nn.Conv1d(128, k, 1))
    def forward(self, x):                                   # x: (batch, 3, points)
        f = self.f1(x)                                      # features of every single point
        g = self.f2(f).max(2, keepdim=True)[0]              # one summary of the whole scene
        g = g.expand(-1, -1, x.shape[2])                    # give the summary to every point
        return self.head(torch.cat([f, g], 1))              # two scores per point

def sample_batch(ids, rng):
    X, Y = [], []
    for i in ids:
        pts, gt, _ = data[i]
        sel = rng.choice(len(pts), N_POINTS, replace=len(pts) < N_POINTS)
        p = pts[sel] / SCALE
        if rng.random() < 0.5:
            p[:, 1] *= -1                                   # mirror left/right as a simple augmentation
        X.append(p.T); Y.append(gt[sel])
    return (torch.tensor(np.stack(X), dtype=torch.float32),
            torch.tensor(np.stack(Y).astype(np.int64)))

model = PointNetSeg()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
rng = np.random.default_rng(0)

print("training...")
t0 = time.time()
for epoch in range(EPOCHS):
    model.train()
    order = rng.permutation(TRAIN_IDS)
    losses = []
    for b in range(0, len(order), BATCH):
        X, Y = sample_batch(order[b:b + BATCH], rng)
        opt.zero_grad()
        loss = loss_fn(model(X), Y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"epoch {epoch + 1}/{EPOCHS}  loss {np.mean(losses):.4f}  ({time.time() - t0:.0f}s)")

os.makedirs("results", exist_ok=True)
torch.save(model.state_dict(), "results/pointnet.pt")

# ---------- testing ----------
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

print("testing on scans", TEST_IDS[0], "to", TEST_IDS[-1])
res_r, res_p = [], []
fa_p = {}
cache = {}
for i in TEST_IDS:
    pts, gt, lab = data[i]
    pr, pp = ransac_road(pts), predict_full(pts)
    cache[i] = (pr, pp)
    res_r.append(iou_f1(pr, gt)); res_p.append(iou_f1(pp, gt))
    cls, cnt = np.unique(lab[pp & ~gt], return_counts=True)
    for k, v in zip(cls, cnt):
        fa_p[int(k)] = fa_p.get(int(k), 0) + int(v)
res_r, res_p = np.array(res_r), np.array(res_p)
print(f"RANSAC   : mean IoU {res_r[:, 0].mean():.3f} | mean F1 {res_r[:, 1].mean():.3f}")
print(f"PointNet : mean IoU {res_p[:, 0].mean():.3f} | mean F1 {res_p[:, 1].mean():.3f}")
tot = sum(fa_p.values())
print("PointNet false alarms by true class:")
for k, v in sorted(fa_p.items(), key=lambda kv: -kv[1])[:5]:
    print(f"  {NAMES.get(k, 'class ' + str(k)):15s} {100 * v / tot:5.1f}%")

# ---------- picture: truth vs RANSAC vs PointNet on one test scan ----------
i = TEST_IDS[len(TEST_IDS) // 2]
pts, gt, _ = data[i]
pr, pp = cache[i]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (title, pred) in zip(axes, [("ground truth", None), ("RANSAC", pr), ("PointNet", pp)]):
    if pred is None:
        colors = np.where(gt[:, None], [[0.0, 0.7, 0.0]], [[0.85, 0.85, 0.85]])
    else:
        colors = np.full((len(pts), 3), 0.85)
        colors[pred & gt] = [0.0, 0.7, 0.0]
        colors[pred & ~gt] = [0.9, 0.0, 0.0]
        colors[~pred & gt] = [0.0, 0.3, 0.9]
    ax.scatter(pts[:, 0], pts[:, 1], c=colors, s=0.4)
    ax.set_aspect("equal"); ax.set_title(f"{title} (scan {i})")
plt.tight_layout()
plt.savefig("results/compare_scan.png", dpi=200)
plt.show()