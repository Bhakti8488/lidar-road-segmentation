import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

SEQ_DIR = "data/dataset/sequences/04"
ROAD_IDS = [40, 44]   # 40 = road, 44 = parking

def load_scan(idx):
    # each scan: groups of 4 numbers (x, y, z, brightness) per point
    pts = np.fromfile(f"{SEQ_DIR}/velodyne/{idx:06d}.bin", dtype=np.float32).reshape(-1, 4)[:, :3]
    # each label: one number per point; the lower 16 bits are the class
    lab = np.fromfile(f"{SEQ_DIR}/labels/{idx:06d}.label", dtype=np.uint32) & 0xFFFF
    is_road = np.isin(lab, ROAD_IDS)      # True = road, False = not road
    return pts, is_road

pts, is_road = load_scan(0)
print("points:", len(pts), "| road points:", int(is_road.sum()))

# Picture 1: bird's-eye view (looking down from above)
os.makedirs("results", exist_ok=True)
plt.figure(figsize=(8, 8))
plt.scatter(pts[:, 0], pts[:, 1], c=is_road, s=0.3, cmap="coolwarm")
plt.axis("equal")
plt.xlabel("x: forward (m)")
plt.ylabel("y: left (m)")
plt.title("SemanticKITTI seq 04, scan 0 (red = road)")
plt.savefig("results/bev_scan0.png", dpi=200)
plt.show()

# Picture 2: interactive 3D view in Open3D
pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
colors = np.full(pts.shape, 0.6)          # grey for everything
colors[is_road] = [0.9, 0.1, 0.1]         # red for road
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd])