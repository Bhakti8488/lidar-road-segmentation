import open3d as o3d
import sys, os
import numpy as np
import torch, torch.nn as nn

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SEQ = sys.argv[2] if len(sys.argv) > 2 else "07"
TAG = sys.argv[3] if len(sys.argv) > 3 else ""
SEQ_DIR = f"data/dataset/sequences/{SEQ}"
ROAD_IDS = [40, 44]
X_RANGE, Y_RANGE = 40, 20
N_POINTS = 4096
SCALE = np.array([X_RANGE, Y_RANGE, 3.0], dtype=np.float32)
o3d.utility.random.seed(0)
rng = np.random.default_rng(0)

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

model = PointNetSeg()
print("loading model file:", f"results/pointnet_{TAG}seed{SEED}.pt")
model.load_state_dict(torch.load(f"results/pointnet_{TAG}seed{SEED}.pt"))
model.eval()

def load_crop(idx):
    pts = np.fromfile(f"{SEQ_DIR}/velodyne/{idx:06d}.bin", dtype=np.float32).reshape(-1, 4)[:, :3]
    lab = np.fromfile(f"{SEQ_DIR}/labels/{idx:06d}.label", dtype=np.uint32) & 0xFFFF
    keep = (np.abs(pts[:, 0]) < X_RANGE) & (np.abs(pts[:, 1]) < Y_RANGE)
    return pts[keep], np.isin(lab, ROAD_IDS)[keep]

def ransac_road(pts):
    cand = np.where(pts[:, 2] < -1.0)[0]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[cand]))
    plane, _ = pcd.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=1000)
    normal = np.array(plane[:3])
    dist = np.abs(pts @ normal + plane[3]) / np.linalg.norm(normal)
    return dist < 0.2

@torch.no_grad()
def predict_full(pts):
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

ids = sorted(int(f[:6]) for f in os.listdir(f"{SEQ_DIR}/velodyne") if f.endswith(".bin"))
print(f"testing model seed {SEED} on sequence {SEQ}: {len(ids)} scans")
res_r, res_p = [], []
for i in ids:
    pts, gt = load_crop(i)
    res_r.append(iou_f1(ransac_road(pts), gt))
    res_p.append(iou_f1(predict_full(pts), gt))
res_r, res_p = np.array(res_r), np.array(res_p)
print(f"RANSAC   : mean IoU {res_r[:, 0].mean():.3f} | mean F1 {res_r[:, 1].mean():.3f}")
print(f"PointNet : mean IoU {res_p[:, 0].mean():.3f} | mean F1 {res_p[:, 1].mean():.3f}")
print(f"RESULT_TRANSFER seed={SEED} seq={SEQ} ransac_iou={res_r[:, 0].mean():.4f} "
      f"pointnet_iou={res_p[:, 0].mean():.4f} pointnet_f1={res_p[:, 1].mean():.4f}")