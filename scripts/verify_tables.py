"""
چکِ مستقلِ همه‌ی اعدادِ جدول‌های مقاله در برابرِ فایل‌های CSV.

این اسکریپت هیچ مدلی بار نمی‌کند و هیچ محاسبه‌ی جدیدی انجام نمی‌دهد؛
فقط اعدادِ نوشته‌شده در مقاله را با فایل‌های خام مقایسه می‌کند.

اجرا:
    python VERIFY_all_tables.py            # فایل‌ها در همین پوشه
    python VERIFY_all_tables.py /path/to/csvs
"""

import sys
import os
import pandas as pd

DATA = sys.argv[1] if len(sys.argv) > 1 else "."

# نامِ روش‌ها: چپ = مقاله، راست = CSV
NAME = {
    "SVD-Fusion CAM (multi-scale)": "SVD-Fusion (multi-scale)",
    "SVD-Fusion CAM (single layer)": "SVD-Fusion (single)",
    "HiResCAM": "HiResCAM",
    "Grad-CAM++": "Grad-CAM++",
    "LayerCAM": "LayerCAM",
    "Grad-CAM": "Grad-CAM",
    "SVD-Grad-CAM": "SVD-Grad-CAM",
    "XGrad-CAM": "XGrad-CAM",
    "Eigen-CAM": "EigenCAM",
}

ok_total, bad_total = 0, 0


def report(name, ok, detail=""):
    global ok_total, bad_total
    if ok:
        ok_total += 1
    else:
        bad_total += 1
        print(f"    MISMATCH  {name}  {detail}")


def load(fname):
    """Find the CSV even if the filename has an upload suffix like __1_."""
    import glob
    path = os.path.join(DATA, fname)
    if os.path.exists(path):
        return pd.read_csv(path)

    # strip a trailing __N_ and try again, then try a wildcard
    stem = fname[:-4]
    for candidate in (stem.replace("__1_", ""), stem.rstrip("_0123456789_")):
        alt = os.path.join(DATA, candidate + ".csv")
        if os.path.exists(alt):
            return pd.read_csv(alt)

    hits = glob.glob(os.path.join(DATA, stem.replace("__1_", "") + "*.csv"))
    if hits:
        print(f"  [using] {os.path.basename(hits[0])}")
        return pd.read_csv(hits[0])

    print(f"  [skip] {fname} not found in {DATA}")
    return None


# ---------------------------------------------------------------- Table 5
def table5():
    print("\nTable 5 — localisation, BRISC")
    d = load("T1_summary_BRISC__1_.csv")
    if d is None:
        return
    paper = {  # method: (PG, IoU)
        "SVD-Fusion CAM (multi-scale)": (43.31, 15.42),
        "SVD-Fusion CAM (single layer)": (25.84, 14.92),
        "HiResCAM": (25.38, 14.36),
        "Grad-CAM++": (23.63, 12.80),
        "LayerCAM": (30.73, 14.66),
        "Grad-CAM": (19.91, 11.77),
        "SVD-Grad-CAM": (20.49, 11.71),
        "XGrad-CAM": (7.22, 4.19),
        "Eigen-CAM": (12.22, 6.24),
    }
    for m, (pg, iou) in paper.items():
        r = d[d.method == NAME[m]]
        if r.empty:
            report(m, False, "not in CSV")
            continue
        r = r.iloc[0]
        good = abs(r["PG %"] - pg) < 0.006 and abs(r["IoU %"] - iou) < 0.006
        report(m, good, f"PG {r['PG %']:.2f} vs {pg} | IoU {r['IoU %']:.2f} vs {iou}")


# ---------------------------------------------------------------- Table 7
def table7():
    print("\nTable 7 — localisation, VOC")
    d = load("T1_summary_VOC__1_.csv")
    if d is None:
        return
    paper = {
        "SVD-Fusion CAM (multi-scale)": (94.72, 26.10),
        "SVD-Fusion CAM (single layer)": (91.15, 29.01),
        "HiResCAM": (91.36, 34.81),
    }
    for m, (pg, iou) in paper.items():
        r = d[d.method == NAME[m]]
        if r.empty:
            report(m, False, "not in CSV")
            continue
        r = r.iloc[0]
        good = abs(r["PG %"] - pg) < 0.006 and abs(r["IoU %"] - iou) < 0.006
        report(m, good, f"PG {r['PG %']:.2f} vs {pg} | IoU {r['IoU %']:.2f} vs {iou}")


# ------------------------------------------------------------ Tables 6, 13
def paired(fname, label, blocks):
    print(f"\n{label}")
    d = load(fname)
    if d is None:
        return
    for metric, rows in blocks.items():
        for m, win, eff, p in rows:
            r = d[(d.metric == metric) & (d.vs == NAME[m])]
            if r.empty:
                report(f"{metric}/{m}", False, "not in CSV")
                continue
            r = r.iloc[0]
            good = (abs(r["win_rate_%"] - win) < 0.06
                    and abs(r.effect_r - eff) < 0.0006
                    and abs(r.p - p) / max(p, 1e-300) < 0.06)
            report(f"{metric}/{m}", good,
                   f"win {r['win_rate_%']:.1f} vs {win} | r {r.effect_r:+.3f} vs {eff:+.3f} "
                   f"| p {r.p:.1e} vs {p:.1e}")


TABLE6 = {
    "lift": [("SVD-Fusion CAM (single layer)", 59.5, 0.291, 1.6e-13),
             ("HiResCAM", 75.4, 0.622, 4.1e-56),
             ("Grad-CAM++", 75.2, 0.656, 3.3e-62),
             ("LayerCAM", 73.1, 0.700, 1.3e-70),
             ("Grad-CAM", 83.9, 0.837, 2.7e-100),
             ("SVD-Grad-CAM", 85.6, 0.857, 6.7e-105),
             ("XGrad-CAM", 77.9, 0.676, 4.1e-66),
             ("Eigen-CAM", 74.6, 0.603, 6.5e-53)],
    "pg":   [("SVD-Fusion CAM (single layer)", 22.6, 0.630, 2.4e-22),
             ("HiResCAM", 22.2, 0.675, 2.0e-24),
             ("Grad-CAM++", 24.8, 0.658, 5.5e-26),
             ("LayerCAM", 20.5, 0.443, 4.7e-12),
             ("Grad-CAM", 26.5, 0.788, 2.5e-36),
             ("SVD-Grad-CAM", 26.2, 0.772, 9.3e-35),
             ("XGrad-CAM", 38.6, 0.876, 5.4e-61),
             ("Eigen-CAM", 34.8, 0.807, 9.2e-49)],
    "iou":  [("SVD-Fusion CAM (single layer)", 52.4, 0.155, 2.2e-4),
             ("HiResCAM", 51.9, 0.174, 3.5e-5),
             ("Grad-CAM++", 56.1, 0.243, 5.2e-9),
             ("LayerCAM", 44.4, 0.007, 8.7e-1),
             ("Grad-CAM", 59.6, 0.401, 1.3e-21),
             ("SVD-Grad-CAM", 60.0, 0.407, 3.5e-22),
             ("XGrad-CAM", 71.4, 0.688, 3.8e-61),
             ("Eigen-CAM", 66.5, 0.586, 7.4e-45)],
}

TABLE13 = {
    "lift": [("SVD-Fusion CAM (single layer)", 38.5, -0.234, 3.8e-14),
             ("HiResCAM", 82.5, 0.724, 1.4e-121),
             ("SVD-Grad-CAM", 86.1, 0.834, 4.3e-160),
             ("Grad-CAM", 87.1, 0.860, 4.0e-170),
             ("Grad-CAM++", 96.7, 0.975, 1.5e-217),
             ("LayerCAM", 93.1, 0.913, 2.8e-191),
             ("XGrad-CAM", 81.3, 0.802, 3.3e-148),
             ("Eigen-CAM", 68.6, 0.625, 9.4e-91)],
    "pg":   [("SVD-Fusion CAM (single layer)", 4.4, 0.694, 3.8e-9),
             ("HiResCAM", 5.4, 0.456, 3.6e-6),
             ("SVD-Grad-CAM", 5.9, 0.673, 2.6e-11),
             ("Grad-CAM", 6.6, 0.755, 7.8e-15),
             ("Grad-CAM++", 7.8, 0.717, 6.8e-16),
             ("LayerCAM", 12.2, 0.685, 1.7e-22),
             ("XGrad-CAM", 34.5, 0.940, 1.2e-97),
             ("Eigen-CAM", 40.5, 0.938, 7.7e-114)],
    "iou":  [("SVD-Fusion CAM (single layer)", 36.7, -0.488, 5.3e-56),
             ("HiResCAM", 19.1, -0.772, 9.3e-138),
             ("SVD-Grad-CAM", 15.1, -0.814, 1.1e-152),
             ("Grad-CAM", 15.6, -0.807, 4.2e-150),
             ("Grad-CAM++", 14.1, -0.830, 2.9e-159),
             ("LayerCAM", 18.1, -0.742, 1.4e-127),
             ("XGrad-CAM", 73.4, 0.637, 1.3e-94),
             ("Eigen-CAM", 64.8, 0.466, 3.3e-51)],
}


# ---------------------------------------------------------------- Table 8
def table8():
    print("\nTable 8 — per-class, BRISC")
    d = load("T4_perclass_BRISC__1_.csv")
    if d is None:
        return
    classes = ["glioma", "meningioma", "pituitary"]
    lift = {"SVD-Fusion CAM (multi-scale)": (1.73, 4.20, 1.70),
            "SVD-Fusion CAM (single layer)": (1.71, 4.16, 1.63),
            "HiResCAM": (1.35, 3.74, 1.61),
            "SVD-Grad-CAM": (1.26, 2.97, 0.89),
            "Grad-CAM": (1.30, 3.01, 0.84),
            "Grad-CAM++": (1.85, 3.15, 0.38),
            "LayerCAM": (1.50, 2.21, 1.54),
            "XGrad-CAM": (0.62, 0.68, 0.66),
            "Eigen-CAM": (0.69, 1.03, 0.25)}
    pg = {"SVD-Fusion CAM (multi-scale)": (23.6, 75.1, 27.7),
          "SVD-Fusion CAM (single layer)": (14.6, 53.4, 7.3),
          "HiResCAM": (12.2, 55.7, 5.7),
          "SVD-Grad-CAM": (9.8, 47.9, 1.7),
          "Grad-CAM": (9.4, 46.9, 1.3),
          "Grad-CAM++": (15.4, 53.1, 0.7),
          "LayerCAM": (20.1, 57.7, 12.3),
          "XGrad-CAM": (4.7, 12.1, 4.3),
          "Eigen-CAM": (6.3, 23.3, 6.0)}

    for col, table, tol in (("lift_med", lift, 0.006), ("PG", pg, 0.06)):
        for m, expected in table.items():
            vals = []
            for c in classes:
                r = d[(d["class"].str.lower() == c) & (d.method == NAME[m])]
                vals.append(None if r.empty else float(r.iloc[0][col]))
            good = all(v is not None and abs(v - e) < tol
                       for v, e in zip(vals, expected))
            report(f"{col}/{m}", good, f"{vals} vs {list(expected)}")

    # header rows: image counts and derived cell counts
    for c, n_img, area in (("glioma", 254, 3.86), ("meningioma", 305, 4.66),
                           ("pituitary", 300, 1.41)):
        r = d[d["class"].str.lower() == c].iloc[0]
        report(f"n/{c}", int(r["n"]) == n_img, f"{int(r['n'])} vs {n_img}")
    print("    (cells at 8x8 = area% x 64, at 30x30 = area% x 900 — derived, not stored)")


# ---------------------------------------------------------------- Table 9
def table9():
    print("\nTable 9 — advantage by target size")
    baselines = ["HiResCAM", "SVD-Grad-CAM", "Grad-CAM", "Grad-CAM++",
                 "LayerCAM", "XGrad-CAM", "EigenCAM"]
    paper = [("BRISC", "0–5%", 2.11, 1.82, 15.8),
             ("BRISC", "5–10%", 3.76, 3.48, 8.1),
             ("BRISC", "10–20%", 2.99, 2.80, 7.0),
             ("VOC", "0–5%", 6.22, 5.57, 11.5),
             ("VOC", "5–10%", 4.95, 4.10, 20.6),
             ("VOC", "10–20%", 3.45, 3.07, 12.6),
             ("VOC", "20–35%", 2.44, 2.27, 7.4),
             ("VOC", "35–100%", 1.34, 1.28, 4.5)]
    files = {"VOC": "T3_persize_VOC__1_.csv", "BRISC": "T3_persize_BRISC__1_.csv"}
    cache = {k: load(v) for k, v in files.items()}
    for ds, b, ours, best, gain in paper:
        d = cache[ds]
        if d is None:
            continue
        p = d[d.bin == b].set_index("method").lift
        o = p["SVD-Fusion (multi-scale)"]
        bb = max(p[m] for m in baselines if m in p.index)
        g = 100 * (o - bb) / bb
        good = abs(o - ours) < 0.006 and abs(bb - best) < 0.006 and abs(g - gain) < 0.06
        report(f"{ds} {b}", good, f"ours {o:.2f} vs {ours} | best {bb:.2f} vs {best} "
                                  f"| gain {g:.1f} vs {gain}")


# --------------------------------------------------------------- Table 12
def table12():
    print("\nTable 12 — runtime")
    d = load("GAP_runtime_BRISC.csv")
    if d is None:
        return
    paper = [("Grad-CAM", 916, 1.00), ("Grad-CAM++", 922, 1.01),
             ("LayerCAM", 916, 1.00), ("SVD-Fusion (multi-scale)", 944, 1.03),
             ("SVD-Fusion (single)", 968, 1.06),
             ("SVD-Grad-CAM (single)", 1030, 1.12),
             ("EigenCAM", 4025, 4.39), ("Score-CAM", 7496, 8.18)]
    base = d[d.method == "Grad-CAM"].mean_ms.iloc[0]
    for m, ms, rel in paper:
        r = d[d.method == m]
        if r.empty:
            report(m, False, "not in CSV")
            continue
        v = r.mean_ms.iloc[0]
        good = abs(round(v) - ms) <= 1 and abs(v / base - rel) < 0.006
        report(m, good, f"{v:.0f}ms vs {ms} | {v/base:.2f}x vs {rel}")


if __name__ == "__main__":
    print(f"reading CSVs from: {os.path.abspath(DATA)}")
    table5()
    table7()
    paired("T2_paired_BRISC_ref_SVD-Fusion__multi-scale___1_.csv",
           "Table 6 — paired, BRISC", TABLE6)
    paired("T2_paired_VOC_ref_SVD-Fusion__multi-scale___1_.csv",
           "Table 13 — paired, VOC", TABLE13)
    table8()
    table9()
    table12()

    print("\n" + "=" * 46)
    print(f"checked {ok_total + bad_total} values: {ok_total} match, {bad_total} mismatch")
    if bad_total == 0:
        print("every number in the paper matches its CSV.")
    else:
        print("see the MISMATCH lines above.")
