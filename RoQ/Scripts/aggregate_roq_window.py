"""
Usage:
  python3 aggregate_roq_window_v2.py \
    --packets ../data/roq_packets_E1.csv \
    --log     ../data/roq_attack_log_E1.csv \
    --out     ../data/dataset_roq_window_E1.csv \
    --W 5 --T 10 --target_ip 10.0.2.15
"""

import argparse
import pandas as pd
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--packets",   required=True)
    p.add_argument("--log",       required=True)
    p.add_argument("--out",       required=True)
    p.add_argument("--W",         type=float, default=5.0,    help="Tamanho janela (s)")
    p.add_argument("--T",         type=float, default=10.0,   help="Período ciclo RoQ (s)")
    p.add_argument("--target_ip", type=str,   default=None,   help="IP do servidor alvo")
    return p.parse_args()

def build_attack_intervals(log_df, T):
    """
    For each cycle, calculate the burst interval:
      t_start = min(ts) of cycle
      t_end   = t_start + max(offset_in_burst) of cycle
    Returns a list of (t_start, t_end)
    """
    intervals = []
    for cycle, grp in log_df.groupby("cycle"):
        t_start = grp["ts"].min()
        tau     = grp["offset_in_burst"].max()
        t_end   = t_start + tau
        intervals.append((t_start, t_end))
    return intervals

def label_window(t_win_start, t_win_end, attack_intervals, overlap_thresh=0.3):
    """
    Window is label=1 if has overlap >= overlap_thresh with any burst.
    Low threshold (0.3) for capture partially overlapping windows using short bursts.
    """
    win_len = t_win_end - t_win_start
    for (a_start, a_end) in attack_intervals:
        overlap = max(0, min(t_win_end, a_end) - max(t_win_start, a_start))
        if overlap / win_len >= overlap_thresh:
            return 1
    return 0

def extract_features(pkts, W):
    n = len(pkts)
    if n < 2:
        return None

    pkt_lens = pkts["frame.len"].values
    iats     = pd.to_numeric(pkts["frame.time_delta_displayed"], errors="coerce").values
    tcp_lens = pd.to_numeric(pkts.get("tcp.len", pd.Series([0]*n)), errors="coerce").fillna(0).values

    bytes_total = float(pkt_lens.sum())
    tcp_bytes   = float(tcp_lens.sum())
    pkts_rate   = n / W
    bytes_rate  = bytes_total / W

    iat_vals  = iats[(iats > 0) & (~np.isnan(iats))]
    iat_mean  = float(iat_vals.mean()) if len(iat_vals) > 0 else 0.0
    iat_std   = float(iat_vals.std())  if len(iat_vals) > 1 else 0.0
    iat_cv    = iat_std / (iat_mean + 1e-9)

    t0 = pkts["frame.time_epoch"].min()
    sub_w = W / 4.0
    sub_pkts  = []
    sub_bytes = []
    for i in range(4):
        s = t0 + i * sub_w
        e = t0 + (i + 1) * sub_w
        mask = (pkts["frame.time_epoch"] >= s) & (pkts["frame.time_epoch"] < e)
        sub_pkts.append(mask.sum())
        sub_bytes.append(float(pkts.loc[mask, "frame.len"].sum()) if mask.any() else 0.0)

    sub_pkts  = np.array(sub_pkts,  dtype=float)
    sub_bytes = np.array(sub_bytes, dtype=float)

    burst_peak_ratio = sub_pkts.max()  / (sub_pkts.mean()  + 1e-9)
    burst_cv         = sub_pkts.std()  / (sub_pkts.mean()  + 1e-9)
    bytes_rate_cv    = sub_bytes.std() / (sub_bytes.mean() + 1e-9)

    return {
        "n_pkts":           n,
        "bytes_total":      bytes_total,
        "tcp_bytes":        tcp_bytes,
        "pkts_rate":        pkts_rate,
        "bytes_rate":       bytes_rate,
        "pkt_size_mean":    float(pkt_lens.mean()),
        "pkt_size_std":     float(pkt_lens.std()) if n > 1 else 0.0,
        "iat_mean":         iat_mean,
        "iat_std":          iat_std,
        "iat_cv":           iat_cv,
        "burst_peak_ratio": burst_peak_ratio,
        "burst_cv":         burst_cv,
        "bytes_rate_cv":    bytes_rate_cv,
        "tcp_ratio":        tcp_bytes / (bytes_total + 1e-9),
    }

def main():
    args = parse_args()
    W = args.W
    T = args.T

    # Load log and construct attack intervals
    print(f"A carregar log: {args.log}")
    log = pd.read_csv(args.log)
    log.columns = [c.strip() for c in log.columns]
    log["ts"]              = pd.to_numeric(log["ts"], errors="coerce")
    log["offset_in_burst"] = pd.to_numeric(log["offset_in_burst"], errors="coerce")
    log = log.dropna(subset=["ts", "offset_in_burst"])

    attack_intervals = build_attack_intervals(log, T)
    tau_mean = np.mean([e - s for s, e in attack_intervals])
    print(f"  {len(attack_intervals)} intervalos de ataque | tau médio={tau_mean:.3f}s")
    print(f"  Exemplo: [{attack_intervals[0][0]:.3f}, {attack_intervals[0][1]:.3f}]")

    # Load packets
    print(f"\nA carregar pacotes: {args.packets}")
    pkts = pd.read_csv(args.packets, dtype=str)
    pkts.columns = [c.strip() for c in pkts.columns]
    pkts["frame.time_epoch"] = pd.to_numeric(pkts["frame.time_epoch"], errors="coerce")
    pkts["frame.len"]        = pd.to_numeric(pkts["frame.len"],        errors="coerce")
    pkts = pkts.dropna(subset=["frame.time_epoch", "frame.len"])
    pkts = pkts.sort_values("frame.time_epoch").reset_index(drop=True)

    # Filter by ip if is given
    if args.target_ip and "ip.dst" in pkts.columns:
        before = len(pkts)
        pkts = pkts[pkts["ip.dst"].str.strip() == args.target_ip.strip()]
        print(f"  Filtrado por ip.dst={args.target_ip}: {before} → {len(pkts)} pacotes")
    else:
        print(f"  {len(pkts)} pacotes totais (sem filtro de IP)")

    t_min = pkts["frame.time_epoch"].min()
    t_max = pkts["frame.time_epoch"].max()
    print(f"  Span: {t_max - t_min:.1f}s | W={W}s | T={T}s")

    # Aggregates in windows
    print(f"\nA agregar em janelas de {W}s...")
    rows = []
    t = t_min
    while t < t_max:
        mask = (pkts["frame.time_epoch"] >= t) & (pkts["frame.time_epoch"] < t + W)
        win_pkts = pkts[mask]
        if len(win_pkts) >= 5:
            feats = extract_features(win_pkts, W)
            if feats:
                feats["label"] = label_window(t, t + W, attack_intervals,
                                               overlap_thresh=0.3)
                rows.append(feats)
        t += W

    df = pd.DataFrame(rows)
    print(f"\n=== RESULTADO ===")
    print(f"Total janelas : {len(df)}")
    print(f"Attack (1)    : {df['label'].eq(1).sum()}")
    print(f"Benign (0)    : {df['label'].eq(0).sum()}")

    if df['label'].eq(1).sum() == 0:
        print("\n  AVISO: Nenhuma janela de ataque detectada!")
        print("   Verifica se o --target_ip está correcto e se os timestamps do log")
        print("   coincidem com os timestamps do pcap.")
        print(f"\n   Log ts range  : [{log['ts'].min():.3f}, {log['ts'].max():.3f}]")
        print(f"   Pkts ts range : [{t_min:.3f}, {t_max:.3f}]")
    else:
        print("\n=== DIAGNÓSTICO (ataque vs benigno) ===")
        key = ["pkts_rate","bytes_rate","burst_peak_ratio","burst_cv","bytes_rate_cv","iat_cv"]
        for feat in key:
            a = df.loc[df["label"]==1, feat].mean()
            b = df.loc[df["label"]==0, feat].mean()
            diff = (a - b) / (abs(b) + 1e-9) * 100
            marker = "" if abs(diff) > 30 else " " if abs(diff) > 10 else ""
            print(f"  {marker} {feat:<20s}: ataque={a:.4f}  benigno={b:.4f}  diff={diff:+.1f}%")

    df.to_csv(args.out, index=False)
    print(f"\nGuardado em: {args.out}")

if __name__ == "__main__":
    main()
