import open3d as o3d
import sys, os
import numpy as np
import torch, torch.nn as nn

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SEQ = sys.argv[2] if len(sys.argv) > 2 else "07"
FIRST = int(sys.argv[3]) if len(sys.argv) > 3 else 0
LAST = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9
SEQ_DIR = f"data/dataset/sequences/{SEQ}"
ROAD_IDS = [40, 44]
X_RANGE, Y_RANGE = 40, 20
N_POINTS = 4096
SCALE = np.array([X_RANGE, Y_RANGE, 3.0], dtype=np.float32)
o3d.utility.random.seed(0)
rng = np.random.default_rng(0)

NAMES = {0: "unlabeled", 1: "outlier", 10: "car", 11: "bicycle", 13: "bus", 15: "motorcycle",
         16: "on-rails", 18: "truck", 20: "other-vehicle", 30: "person", 31: "bicyclist",
         32: "motorcyclist", 40: "road", 44: "parking", 48: "sidewalk", 49: "other-ground",
         50: "building", 51: "fence", 52: "other-structure", 60: "lane-marking",
         70: "vegetation", 71: "trunk", 72: "terrain", 80: "pole", 81: "traffic-sign",
         99: "other-object", 252: "moving-car", 253: "moving-bicyclist", 254: "moving-person",
         255: "moving-motorcyclist", 256: "moving-on-rails", 257: "moving-bus",
         258: "moving-truck", 259: "moving-other-vehicle"}

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
model.load_state_dict(torch.load(f"results/pointnet_seed{SEED}.pt"))
model.eval()

def load_crop(idx):
    pts = np.fromfile(f"{SEQ_DIR}/velodyne/{idx:06d}.bin", dtype=np.float32).reshape(-1, 4)[:, :3]
    lab = np.fromfile(f"{SEQ_DIR}/labels/{idx:06d}.label", dtype=np.uint32) & 0xFFFF
    keep = (np.abs(pts[:, 0]) < X_RANGE) & (np.abs(pts[:, 1]) < Y_RANGE)
    return pts[keep], lab[keep]

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
ids = [i for i in ids if FIRST <= i <= LAST]
print(f"model seed {SEED} | sequence {SEQ} | scans {ids[0]}-{ids[-1]} ({len(ids)} scans)")

dist, lat, labs, gts, pp, pr = [], [], [], [], [], []
per_scan_p, per_scan_r = [], []
for i in ids:
    pts, lab = load_crop(i)
    gt = np.isin(lab, ROAD_IDS)
    p_pred, r_pred = predict_full(pts), ransac_road(pts)
    per_scan_p.append(iou_f1(p_pred, gt)[0]); per_scan_r.append(iou_f1(r_pred, gt)[0])
    dist.append(np.linalg.norm(pts[:, :2], axis=1)); lat.append(np.abs(pts[:, 1]))
    labs.append(lab); gts.append(gt); pp.append(p_pred); pr.append(r_pred)
dist, lat, labs = np.concatenate(dist), np.concatenate(lat), np.concatenate(labs)
gts, pp, pr = np.concatenate(gts), np.concatenate(pp), np.concatenate(pr)

print(f"sanity (per-scan mean IoU): RANSAC {np.mean(per_scan_r):.3f} | PointNet {np.mean(per_scan_p):.3f}")
print(f"pooled IoU:                 RANSAC {iou_f1(pr, gts)[0]:.3f} | PointNet {iou_f1(pp, gts)[0]:.3f}")

def report(title, vals, edges, unit):
    print(f"\nIoU by {title} (pooled)")
    for a, b in zip(edges[:-1], edges[1:]):
        m = (vals >= a) & (vals < b)
        share = 100 * gts[m].sum() / max(gts.sum(), 1)
        print(f"  {a:>3.0f}-{b:<3.0f}{unit}: RANSAC {iou_f1(pr[m], gts[m])[0]:.3f} | "
              f"PointNet {iou_f1(pp[m], gts[m])[0]:.3f} | share of road points {share:4.1f}%")

report("distance from the car", dist, [0, 10, 20, 30, 45], " m")
report("sideways distance |y|", lat, [0, 3, 6, 10, 20.01], " m")

print("\nRecall (share of true points found as road)")
for cid in ROAD_IDS:
    m = labs == cid
    if m.sum():
        print(f"  {NAMES[cid]:8s}: RANSAC {pr[m].mean():.3f} | PointNet {pp[m].mean():.3f} ({m.sum()} points)")

for name, pred in [("PointNet", pp), ("RANSAC", pr)]:
    fa = pred & ~gts
    cls, cnt = np.unique(labs[fa], return_counts=True)
    order = np.argsort(-cnt)[:5]
    print(f"\n{name} false alarms by true class")
    for k in order:
        print(f"  {NAMES.get(int(cls[k]), 'class ' + str(int(cls[k]))):15s} {100 * cnt[k] / cnt.sum():5.1f}%")