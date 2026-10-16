"""Generate synthetic network flow records with injected anomalies.

Normal traffic is drawn from three behaviour clusters (web browsing, video
streaming, DNS). Anomalies (~5%) simulate a port scan and a data
exfiltration event. Output: data/flows.csv with a ground-truth `label`
column (0 = normal, 1 = anomaly).
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N_FLOWS = 5000
ANOMALY_RATE = 0.05

OUT = Path(__file__).resolve().parent.parent / "data" / "flows.csv"


def normal_flows(n):
    """Three clusters of benign traffic."""
    n_each = n // 3
    # Web browsing: short flows, small transfers, ports 80/443
    web = pd.DataFrame({
        "duration_s": RNG.exponential(2.0, n_each),
        "bytes": RNG.lognormal(8, 1.0, n_each).astype(int),
        "packets": RNG.poisson(12, n_each),
        "dst_port": RNG.choice([80, 443], n_each),
        "protocol": "tcp",
    })
    # Video streaming: long flows, large transfers
    video = pd.DataFrame({
        "duration_s": RNG.exponential(120.0, n_each),
        "bytes": RNG.lognormal(15, 1.0, n_each).astype(int),
        "packets": RNG.poisson(800, n_each),
        "dst_port": RNG.choice([80, 443, 1935], n_each),
        "protocol": "tcp",
    })
    # DNS: tiny UDP flows on port 53
    dns = pd.DataFrame({
        "duration_s": RNG.exponential(0.05, n - 2 * n_each),
        "bytes": RNG.poisson(180, n - 2 * n_each),
        "packets": RNG.poisson(2, n - 2 * n_each),
        "dst_port": 53,
        "protocol": "udp",
    })
    return pd.concat([web, video, dns], ignore_index=True)


def port_scan(n):
    """Many short flows sweeping sequential destination ports."""
    return pd.DataFrame({
        "duration_s": RNG.exponential(0.02, n),
        "bytes": RNG.poisson(60, n),
        "packets": RNG.poisson(1, n),
        "dst_port": np.arange(1, n + 1) % 1024,
        "protocol": "tcp",
    })


def exfiltration(n):
    """A few very large outbound transfers on an unusual port."""
    return pd.DataFrame({
        "duration_s": RNG.exponential(300.0, n),
        "bytes": RNG.lognormal(20, 0.5, n).astype(int),
        "packets": RNG.poisson(5000, n),
        "dst_port": 4444,
        "protocol": "tcp",
    })


def main():
    n_anom = int(N_FLOWS * ANOMALY_RATE)
    n_scan = n_anom // 2
    n_exfil = n_anom - n_scan

    normal = normal_flows(N_FLOWS - n_anom).assign(label=0)
    scan = port_scan(n_scan).assign(label=1)
    exfil = exfiltration(n_exfil).assign(label=1)

    flows = pd.concat([normal, scan, exfil], ignore_index=True)
    flows = flows.sample(frac=1.0, random_state=42).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    flows.to_csv(OUT, index=False)
    print(f"Wrote {len(flows)} flows ({flows['label'].sum()} anomalies) to {OUT}")


if __name__ == "__main__":
    main()
