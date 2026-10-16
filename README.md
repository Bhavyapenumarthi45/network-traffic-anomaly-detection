# Network Traffic Anomaly Detection

A small, self-contained demo that detects anomalous network flows with an
Isolation Forest model. Because real packet captures can't be shared in a
portfolio repo, the traffic here is **synthetically generated** — but the
feature engineering and detection pipeline mirror how a real flow-based
anomaly detector works.

## How it works

1. `src/generate_data.py` synthesizes ~5,000 network flow records
   (duration, bytes, packets, ports, protocol) with ~5% injected anomalies:
   - **Port scans** — many short flows sweeping sequential destination ports
   - **Data exfiltration** — unusually large outbound transfers
2. `src/detect.py` trains an Isolation Forest on flow features and evaluates
   it against the known labels (precision, recall, F1).

## Quickstart

```bash
pip install -r requirements.txt
python src/generate_data.py   # creates data/flows.csv
python src/detect.py          # trains, evaluates, saves data/anomalies.png
```

Expected output is an F1 score around 0.72 on the synthetic data.

## Files

| File | Purpose |
| ---- | ------- |
| `src/generate_data.py` | Synthetic flow generator (normal traffic + injected attacks) |
| `src/detect.py` | Feature engineering, Isolation Forest training, evaluation |
| `requirements.txt` | Python dependencies |
| `data/flows.csv` | Generated dataset (created on first run, git-ignored) |

## Extending it

Swap `generate_data.py` for a parser that reads real NetFlow/IPFIX exports
or Zeek logs — the feature set in `detect.py` (`duration`, `bytes`,
`packets`, `dst_port`, `protocol`) maps directly onto flow records.
