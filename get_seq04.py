from remotezip import RemoteZip

URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip"
SEQ = "04"

with RemoteZip(URL) as z:
    all_names = z.namelist()
    print("first entries:", all_names[:3])
    names = [n for n in all_names
             if f"/sequences/{SEQ}/velodyne/" in n and n.endswith(".bin")]
    print(len(names), "files found")
    for i, n in enumerate(names):
        z.extract(n, "data")
        if i % 50 == 0:
            print(i, "done")
print("finished")