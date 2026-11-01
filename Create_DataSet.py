import pandas as pd
import numpy as np
import os

np.random.seed(42)

n = 600

labels = np.random.choice(
    ["BENIGN", "DDoS", "PortScan", "DoS Hulk",
     "FTP-Patator", "SSH-Patator"],
    n,
    p=[0.55, 0.12, 0.10, 0.10, 0.07, 0.06]
)

attack = labels != "BENIGN"
ddos = labels == "DDoS"
portscan = labels == "PortScan"
dos = labels == "DoS Hulk"
brute = np.isin(labels, ["FTP-Patator", "SSH-Patator"])

duration = np.random.lognormal(9, 1, n)
duration[ddos] *= 0.25
duration[portscan] *= 0.15
duration[brute] *= 1.8

fwd = np.random.poisson(18, n) + 1
bwd = np.random.poisson(15, n) + 1

fwd[ddos | dos] += np.random.poisson(100, (ddos | dos).sum())
bwd[ddos | dos] += np.random.poisson(70, (ddos | dos).sum())

fwd[portscan] = np.random.randint(1, 5, portscan.sum())
bwd[portscan] = np.random.randint(1, 3, portscan.sum())

fwd_bytes = fwd * np.random.randint(40, 1400, n)
bwd_bytes = bwd * np.random.randint(40, 1400, n)

fwd_bytes[ddos | dos] *= 3
bwd_bytes[ddos | dos] *= 2

syn = np.random.poisson(2, n)
syn[ddos | dos | portscan] += np.random.randint(
    10, 80, (ddos | dos | portscan).sum()
)

df = pd.DataFrame({
    "Flow Duration": duration.round(2),
    "Total Fwd Packets": fwd,
    "Total Backward Packets": bwd,
    "Total Length of Fwd Packets": fwd_bytes,
    "Total Length of Bwd Packets": bwd_bytes,
    "Fwd Packet Length Mean": (fwd_bytes / fwd).round(2),
    "Bwd Packet Length Mean": (bwd_bytes / bwd).round(2),
    "Flow Bytes/s": (
        (fwd_bytes + bwd_bytes) /
        np.maximum(duration / 1000000, 0.001)
    ).round(2),
    "Flow Packets/s": (
        (fwd + bwd) /
        np.maximum(duration / 1000000, 0.001)
    ).round(2),
    "SYN Flag Count": syn,
    "ACK Flag Count": np.random.poisson(12, n) + attack * 2,
    "Destination Port": np.random.choice(
        [21, 22, 23, 53, 80, 443, 445, 3389, 8080],
        n
    ),
    "Label": labels
})

os.makedirs("data", exist_ok=True)

df.to_csv("data/network_traffic_sample.csv", index=False)

print("Dataset created successfully!")
print("Total records:", len(df))
print("Total columns:", len(df.columns))
print(df["Label"].value_counts())
