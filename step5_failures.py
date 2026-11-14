import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

o3d.utility.random.seed(0)
SEQ_DIR = "data/dataset/sequences/04"
ROAD_IDS = [40, 44]          # 40 = road, 44 = parking
X_RANGE, Y_RANGE = 40, 20
DIST_THR = 0.2

NAMES = {0: "unlabeled", 1: "outlier", 10: "car", 11: "bicycle", 13: "bus", 15: "motorcycle",
         16: "on-rails", 18: "truck", 20: "other-vehicle", 30: "person", 31: "bicyclist",
         32: "motorcyclist", 40: "road", 44: "parking", 48: "sidewalk", 49: "other-ground",
         50: "building", 51: "fence", 52: "other-structure", 60: "lane-marking",
         70: "vegetation", 71: "trunk", 72: "terrain", 80: "pole", 81: "traffic-sign",
         99: "other-object"}

def load_raw(idx):
    pts = np.fromfile(f"{SEQ_DIR}/velodyne/{idx:06d}.bin", dtype=np.float32).reshape(-1, 4)[:, :3]
    lab = np.fromfile(f"{SEQ_DIR}/labels/{idx:06d}.label", dtype=np.uint32) & 0xFFFF
    return pts, lab

def ransac_road(pts):
    in_crop = (np.abs(pts[:, 0]) < X_RANGE) & (np.abs(pts[:, 1]) < Y_RANGE)
    cand = np.where(in_crop & (pts[:, 2] < -1.0))[0]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[cand]))
    plane, _ = pcd.segment_plane(distance_threshold=DIST_THR, ransac_n=3, num_iterations=1000)
    normal = np.array(plane[:3])
    dist = np.abs(pts @ normal + plane[3]) / np.linalg.norm(normal)
    return in_crop & (dist < DIST_THR), in_crop

n_scans = len(os.listdir(f"{SEQ_DIR}/velodyne"))
fa_counts, ious = {}, []
r_hit, r_miss = [], []

for i in range(n_scans):
    pts, lab = load_raw(i)
    gt = np.isin(lab, ROAD_IDS)
    pred, in_crop = ransac_road(pts)
    tp, fp, fn = pred & gt, pred & ~gt, ~pred & gt & in_crop
    ious.append(tp.sum() / (tp.sum() + fp.sum() + fn.sum() + 1e-9))
    classes, counts = np.unique(lab[fp], return_counts=True)
    for k, v in zip(classes, counts):
        fa_counts[int(k)] = fa_counts.get(int(k), 0) + int(v)
    rng = np.linalg.norm(pts[:, :2], axis=1)
    r_hit.append(rng[tp])
    r_miss.append(rng[fn])

total = sum(fa_counts.values())
print("False alarms by true class:")
for k, v in sorted(fa_counts.items(), key=lambda kv: -kv[1])[:6]:
    print(f"  {NAMES.get(k, 'class ' + str(k)):15s} {100 * v / total:5.1f}%")
print(f"median distance of correctly found road points: {np.median(np.concatenate(r_hit)):.1f} m")
print(f"median distance of missed road points:          {np.median(np.concatenate(r_miss)):.1f} m")

# Picture of the 3 worst scans
os.makedirs("results", exist_ok=True)
worst = np.argsort(ious)[:3]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, i in zip(axes, worst):
    pts, lab = load_raw(int(i))
    gt = np.isin(lab, ROAD_IDS)
    pred, in_crop = ransac_road(pts)
    p, g, q = pts[in_crop], gt[in_crop], pred[in_crop]
    colors = np.full((len(p), 3), 0.85)
    colors[q & g] = [0.0, 0.7, 0.0]
    colors[q & ~g] = [0.9, 0.0, 0.0]
    colors[~q & g] = [0.0, 0.3, 0.9]
    ax.scatter(p[:, 0], p[:, 1], c=colors, s=0.4)
    ax.set_aspect("equal")
    ax.set_title(f"scan {int(i)}  IoU {ious[int(i)]:.2f}")
plt.tight_layout()
plt.savefig("results/worst3.png", dpi=200)
plt.show()