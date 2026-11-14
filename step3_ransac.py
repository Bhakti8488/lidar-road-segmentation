import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

o3d.utility.random.seed(0)
SEQ_DIR = "data/dataset/sequences/04"
ROAD_IDS = [40, 44]          # 40 = road, 44 = parking
X_RANGE, Y_RANGE = 40, 20    # area around the car we evaluate (meters)
DIST_THR = 0.2               # points within 20 cm of the plane count as ground

def load_scan(idx):
    pts = np.fromfile(f"{SEQ_DIR}/velodyne/{idx:06d}.bin", dtype=np.float32).reshape(-1, 4)[:, :3]
    lab = np.fromfile(f"{SEQ_DIR}/labels/{idx:06d}.label", dtype=np.uint32) & 0xFFFF
    return pts, np.isin(lab, ROAD_IDS)

def ransac_road(pts):
    in_crop = (np.abs(pts[:, 0]) < X_RANGE) & (np.abs(pts[:, 1]) < Y_RANGE)
    cand = np.where(in_crop & (pts[:, 2] < -1.0))[0]       # points allowed to vote
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[cand]))
    plane, _ = pcd.segment_plane(distance_threshold=DIST_THR, ransac_n=3, num_iterations=1000)
    normal = np.array(plane[:3])
    dist = np.abs(pts @ normal + plane[3]) / np.linalg.norm(normal)
    pred = in_crop & (dist < DIST_THR)                      # True = predicted road
    return pred, in_crop

def iou_f1(pred, gt):
    tp = np.sum(pred & gt)       # said road, is road
    fp = np.sum(pred & ~gt)      # said road, is not (false alarm)
    fn = np.sum(~pred & gt)      # said not road, is road (missed)
    iou = tp / (tp + fp + fn + 1e-9)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
    return iou, f1

# ---- Picture for scan 0: green = correct road, red = false alarm, blue = missed road
os.makedirs("results", exist_ok=True)
pts, gt = load_scan(0)
pred, in_crop = ransac_road(pts)
p, g, q = pts[in_crop], gt[in_crop], pred[in_crop]
colors = np.full((len(p), 3), 0.85)
colors[q & g] = [0.0, 0.7, 0.0]
colors[q & ~g] = [0.9, 0.0, 0.0]
colors[~q & g] = [0.0, 0.3, 0.9]
plt.figure(figsize=(9, 5))
plt.scatter(p[:, 0], p[:, 1], c=colors, s=0.5)
plt.axis("equal")
plt.title("RANSAC baseline, scan 0 (green=correct, red=false alarm, blue=missed)")
plt.savefig("results/bev_ransac_scan0.png", dpi=200)
plt.show()

# ---- Score all scans
n_scans = len(os.listdir(f"{SEQ_DIR}/velodyne"))
scores = []
for i in range(n_scans):
    pts, gt = load_scan(i)
    pred, in_crop = ransac_road(pts)
    scores.append(iou_f1(pred[in_crop], gt[in_crop]))
scores = np.array(scores)
print(f"scans: {n_scans}")
print(f"mean IoU: {scores[:, 0].mean():.3f} | mean F1: {scores[:, 1].mean():.3f}")
worst = np.argsort(scores[:, 0])[:3]
print("3 worst scans:", worst.tolist(), "with IoU", scores[worst, 0].round(3).tolist())