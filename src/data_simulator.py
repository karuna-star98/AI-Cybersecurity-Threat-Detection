"""
Virtual simulation of network traffic.

We do not have a corporate network, so we generate flow-level records that look
like the features in public IDS datasets (NSL-KDD / CICIDS2017 / UNSW-NB15).
Each attack class has a realistic statistical "fingerprint":

  Normal            - moderate bytes/packets, successful sessions, low error rates
  DoS               - huge request rate, many half-open connections (SYN flood)
  BruteForce        - repeated failed logins against ssh/ftp/telnet
  PortScan          - tiny packets to MANY different ports, many rejected connections
  DataExfiltration  - very large outbound transfer, long duration, often at night

~6% of attack rows are made to look like normal traffic ("stealthy" attacks),
and ~1% of values are missing, so the data is realistically messy and the
model cannot reach a fake 100% score.
"""
import numpy as np
import pandas as pd

from .config import CLASSES, NUMERIC, CATEGORICAL

SERVICES = ["http", "https", "ftp", "ssh", "dns", "smtp", "telnet", "other"]


def _normal(rng, n):
    return pd.DataFrame({
        "duration": rng.exponential(2.0, n),
        "protocol_type": rng.choice(["tcp", "udp", "icmp"], n, p=[0.70, 0.25, 0.05]),
        "service": rng.choice(SERVICES, n, p=[0.30, 0.25, 0.04, 0.06, 0.20, 0.07, 0.01, 0.07]),
        "flag": rng.choice(["SF", "S0", "REJ", "RSTO"], n, p=[0.93, 0.03, 0.03, 0.01]),
        "src_bytes": rng.lognormal(6.5, 1.2, n),
        "dst_bytes": rng.lognormal(7.5, 1.5, n),
        "packet_count": rng.poisson(20, n) + 1,
        "failed_logins": rng.binomial(1, 0.04, n) * rng.integers(1, 3, n),
        "logged_in": rng.binomial(1, 0.85, n),
        "request_frequency": rng.gamma(2.0, 5.0, n),
        "count": rng.poisson(8, n),
        "srv_count": rng.poisson(6, n),
        "serror_rate": rng.beta(1, 30, n),
        "rerror_rate": rng.beta(1, 30, n),
        "same_srv_rate": rng.beta(8, 2, n),
        "unique_dst_ports": rng.poisson(2, n) + 1,
    })


def _dos(rng, n):
    return pd.DataFrame({
        "duration": rng.exponential(0.1, n),
        "protocol_type": rng.choice(["tcp", "udp", "icmp"], n, p=[0.60, 0.15, 0.25]),
        "service": rng.choice(["http", "https", "dns", "other"], n, p=[0.5, 0.2, 0.1, 0.2]),
        "flag": rng.choice(["S0", "SF", "REJ"], n, p=[0.70, 0.20, 0.10]),
        "src_bytes": rng.lognormal(5.0, 1.0, n),
        "dst_bytes": rng.lognormal(1.0, 1.5, n),
        "packet_count": rng.poisson(300, n) + 1,
        "failed_logins": np.zeros(n, dtype=int),
        "logged_in": np.zeros(n, dtype=int),
        "request_frequency": rng.gamma(5.0, 80.0, n),
        "count": rng.poisson(150, n),
        "srv_count": rng.poisson(140, n),
        "serror_rate": rng.beta(15, 2, n),
        "rerror_rate": rng.beta(1, 20, n),
        "same_srv_rate": rng.beta(9, 1, n),
        "unique_dst_ports": rng.integers(1, 3, n),
    })


def _bruteforce(rng, n):
    return pd.DataFrame({
        "duration": rng.exponential(3.0, n),
        "protocol_type": np.full(n, "tcp"),
        "service": rng.choice(["ssh", "ftp", "telnet", "http"], n, p=[0.55, 0.20, 0.15, 0.10]),
        "flag": rng.choice(["SF", "REJ", "RSTO"], n, p=[0.55, 0.30, 0.15]),
        "src_bytes": rng.lognormal(5.5, 0.6, n),
        "dst_bytes": rng.lognormal(5.5, 0.6, n),
        "packet_count": rng.poisson(30, n) + 1,
        "failed_logins": rng.poisson(6, n) + 2,
        "logged_in": rng.binomial(1, 0.08, n),
        "request_frequency": rng.gamma(3.0, 15.0, n),
        "count": rng.poisson(25, n),
        "srv_count": rng.poisson(24, n),
        "serror_rate": rng.beta(2, 15, n),
        "rerror_rate": rng.beta(6, 10, n),
        "same_srv_rate": rng.beta(9, 1, n),
        "unique_dst_ports": rng.integers(1, 3, n),
    })


def _portscan(rng, n):
    return pd.DataFrame({
        "duration": rng.exponential(0.05, n),
        "protocol_type": rng.choice(["tcp", "udp"], n, p=[0.85, 0.15]),
        "service": rng.choice(SERVICES, n),
        "flag": rng.choice(["REJ", "RSTO", "S0"], n, p=[0.55, 0.25, 0.20]),
        "src_bytes": rng.exponential(30, n),
        "dst_bytes": np.zeros(n),
        "packet_count": rng.poisson(3, n) + 1,
        "failed_logins": np.zeros(n, dtype=int),
        "logged_in": np.zeros(n, dtype=int),
        "request_frequency": rng.gamma(3.0, 20.0, n),
        "count": rng.poisson(60, n),
        "srv_count": rng.poisson(3, n),
        "serror_rate": rng.beta(4, 6, n),
        "rerror_rate": rng.beta(10, 3, n),
        "same_srv_rate": rng.beta(1, 8, n),
        "unique_dst_ports": rng.integers(20, 500, n),
    })


def _exfil(rng, n):
    return pd.DataFrame({
        "duration": rng.exponential(60.0, n) + 20,
        "protocol_type": np.full(n, "tcp"),
        "service": rng.choice(["https", "ftp", "smtp", "other"], n, p=[0.45, 0.25, 0.15, 0.15]),
        "flag": np.full(n, "SF"),
        "src_bytes": rng.lognormal(13.0, 1.0, n),
        "dst_bytes": rng.lognormal(6.0, 1.0, n),
        "packet_count": rng.poisson(900, n) + 1,
        "failed_logins": np.zeros(n, dtype=int),
        "logged_in": np.ones(n, dtype=int),
        "request_frequency": rng.gamma(2.0, 3.0, n),
        "count": rng.poisson(4, n),
        "srv_count": rng.poisson(4, n),
        "serror_rate": rng.beta(1, 40, n),
        "rerror_rate": rng.beta(1, 40, n),
        "same_srv_rate": rng.beta(9, 1, n),
        "unique_dst_ports": rng.integers(1, 3, n),
    })


_GENERATORS = {
    "Normal": _normal, "DoS": _dos, "BruteForce": _bruteforce,
    "PortScan": _portscan, "DataExfiltration": _exfil,
}


def _hour_probs(night):
    p = np.ones(24)
    if night:
        p[0:6] = 6.0          # 00:00-05:59 heavy
    else:
        p[8:19] = 4.0         # office hours heavy
    return p / p.sum()


def generate_traffic(n_samples=20000, attack_ratio=0.30, seed=42,
                     start="2026-01-01", messy=True):
    """Create a labelled network-traffic DataFrame.

    Parameters
    ----------
    n_samples    : total number of flow records
    attack_ratio : share of records that are attacks (rest is Normal)
    seed         : random seed (same seed -> same data, good for reproducibility)
    messy        : add missing values + duplicate rows to practise cleaning
    """
    rng = np.random.default_rng(seed)
    weights = np.array([0.40, 0.25, 0.20, 0.15])           # DoS, Brute, Scan, Exfil
    n_attack = int(n_samples * attack_ratio)
    counts = dict(zip(CLASSES[1:], (weights * n_attack).astype(int)))
    counts["Normal"] = n_samples - sum(counts.values())

    frames = []
    for label in CLASSES:
        part = _GENERATORS[label](rng, counts[label])
        part["attack_type"] = label
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)

    # "Stealthy" attacks: copy normal-looking numeric values into ~6% of attack rows
    attack_idx = df.index[df["attack_type"] != "Normal"].to_numpy()
    stealth = rng.choice(attack_idx, size=int(0.06 * len(attack_idx)), replace=False)
    normal_like = _normal(rng, len(stealth))
    for col in NUMERIC:
        if col in normal_like.columns:
            df.loc[stealth, col] = normal_like[col].to_numpy()

    # Timestamps over 7 days; brute force / exfiltration happen more at night
    base = pd.Timestamp(start)
    day = rng.integers(0, 7, len(df))
    daytime = rng.choice(24, len(df), p=_hour_probs(night=False))
    nighttime = rng.choice(24, len(df), p=_hour_probs(night=True))
    night_attack = df["attack_type"].isin(["BruteForce", "DataExfiltration"]).to_numpy()
    hour = np.where(night_attack & (rng.random(len(df)) < 0.7), nighttime, daytime)
    secs = rng.integers(0, 3600, len(df))
    df["timestamp"] = (base + pd.to_timedelta(day, unit="D")
                       + pd.to_timedelta(hour * 3600 + secs, unit="s"))
    df["hour"] = df["timestamp"].dt.hour

    # IP addresses: normal users are spread out, attackers come from a few hostile IPs
    df["dst_ip"] = ["10.0.%d.%d" % (rng.integers(0, 4), rng.integers(1, 60)) for _ in range(len(df))]
    attackers = {a: ["203.0.113.%d" % i for i in range(k, k + 4)]
                 for a, k in zip(CLASSES[1:], (10, 20, 30, 40))}
    df["src_ip"] = [
        "192.168.%d.%d" % (rng.integers(0, 10), rng.integers(2, 250)) if a == "Normal"
        else str(rng.choice(attackers[a]))
        for a in df["attack_type"]
    ]

    df = df.sort_values("timestamp").reset_index(drop=True)

    for col in ["failed_logins", "logged_in", "packet_count", "count", "srv_count", "unique_dst_ports"]:
        df[col] = df[col].round().astype(int)
    if messy:
        for col in ["src_bytes", "dst_bytes", "duration", "request_frequency"]:
            df.loc[rng.random(len(df)) < 0.01, col] = np.nan
        dupes = df.sample(frac=0.005, random_state=seed)
        df = pd.concat([df, dupes]).sort_values("timestamp").reset_index(drop=True)

    cols = ["timestamp", "src_ip", "dst_ip"] + CATEGORICAL + NUMERIC + ["attack_type"]
    return df[cols]
