# Architecture

## Data flow
```
 data_simulator.py ──► data/network_traffic.csv ──► preprocessing.clean_data
                                                          │
                                       features.build_preprocessor (inside every model)
                          engineer features → impute → scale numbers / one-hot categories
                                                          │
                       ┌──────────────────────────────────┼───────────────────────────┐
                 Logistic Regression              Random Forest (main)         Isolation Forest
                 (baseline)                       attack type + probability    trained on NORMAL only
                                                          │                          │
                                                          └────────────┬─────────────┘
                                                              detect.score_traffic
                                                                       │
                                          detect.build_alerts: severity + reason per alert
                                                                       │
                                 outputs/alerts.csv|json|log   images/*.png   REST API /predict
```

## Modules
| File | Responsibility |
|---|---|
| `data_simulator.py` | Generates labelled flows with attack fingerprints, timestamps, IPs, missing values |
| `preprocessing.py` | Loads CSV, validates columns, drops duplicates, fixes impossible values |
| `features.py` | Engineered features (`bytes_ratio`, `login_fail_ratio`, `is_night`, …) + sklearn pipeline |
| `train.py` | Stratified 80/20 split, trains 3 models, saves models/metrics/test predictions |
| `evaluate.py` | Accuracy, precision, recall, F1, confusion matrix |
| `detect.py` | Combines both detectors; severity rules; human-readable alert reasons |
| `visualize.py` | 10 charts incl. the SOC dashboard |
| `app.py` | Flask API: `/health`, `/predict`, `/predict_batch` |

## Why two detectors?
* **Random Forest** learns known attack patterns → tells you *what* attack it is.
* **Isolation Forest** learns only normal behaviour → flags anything strange, so it can catch attack types the classifier never saw (alerts labelled `UnknownAnomaly`, threshold = 99th percentile of normal-traffic scores).

## Severity rules
`DataExfiltration`=Critical, `DoS`/`BruteForce`=High, `PortScan`/`UnknownAnomaly`=Medium; lowered one level if model confidence < 0.60.

## Using your own dataset
CSV columns required: `protocol_type, service, flag, duration, src_bytes, dst_bytes, packet_count, failed_logins, logged_in, request_frequency, count, srv_count, serror_rate, rerror_rate, same_srv_rate, unique_dst_ports, hour, timestamp, attack_type` (labels: Normal, DoS, BruteForce, PortScan, DataExfiltration). Map public-dataset columns (NSL-KDD, CICIDS2017) to these names first.
