import sys
from remotezip import RemoteZip

SEQ = sys.argv[1] if len(sys.argv) > 1 else "07"      # which sequence
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300    # how many scans
URL = open("kitti_url.txt").read().strip()            # link stays in your local file

with RemoteZip(URL) as z:
    names = sorted(n for n in z.namelist()
                   if f"/sequences/{SEQ}/velodyne/" in n and n.endswith(".bin"))
    print(len(names), "scans in sequence", SEQ, "- taking", min(N, len(names)))
    for i, n in enumerate(names[:N]):
        z.extract(n, "data")
        if i % 50 == 0:
            print(i, "done")
print("finished")  