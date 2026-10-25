# -*- coding: utf-8 -*-
"""
Lift by target-size bin (appendix figure of the paper).

Input:  T3_persize_VOC.csv, T3_persize_BRISC.csv
Output: FIG_per_size.pdf (and a .png preview)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- paths (edit if needed) ----
VOC_CSV   = 'T3_persize_VOC.csv'
BRISC_CSV = 'T3_persize_BRISC.csv'
OUT_PDF   = 'FIG_per_size.pdf'
OUT_PNG   = 'FIG_per_size.png'

# method order and display labels
METHODS = [
    ('SVD-Fusion (multi-scale)', 'Ours (multi-scale)'),
    ('SVD-Fusion (single)',      'Ours (single layer)'),
    ('HiResCAM',                 'HiResCAM'),
    ('SVD-Grad-CAM',             'SVD-Grad-CAM'),
    ('Grad-CAM',                 'Grad-CAM'),
    ('Grad-CAM++',               'Grad-CAM++'),
    ('LayerCAM',                 'LayerCAM'),
    ('XGrad-CAM',                'XGrad-CAM'),
    ('EigenCAM',                 'Eigen-CAM'),
]
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

BINS_VOC   = ['0–5%', '5–10%', '10–20%', '20–35%', '35–100%']
BINS_BRISC = ['0–5%', '5–10%', '10–20%']


def load(path, order):
    d = pd.read_csv(path)
    p = d.pivot(index='method', columns='bin', values='lift')
    return p[order]


def draw(ax, table, bins, title):
    n = len(METHODS)
    x = np.arange(len(bins))
    width = 0.9 / n
    for i, ((key, _label), colour) in enumerate(zip(METHODS, COLORS)):
        vals = [table.loc[key, b] if key in table.index else np.nan for b in bins]
        ax.bar(x + (i - (n - 1) / 2) * width, vals, width,
               color=colour, edgecolor='none')
    ax.axhline(1.0, color='0.45', linestyle=':', linewidth=1.0, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_xlabel('target size bin')
    ax.set_ylabel('median lift (EBPG $\\div$ chance)')
    ax.set_title(title)
    ax.grid(axis='y', color='0.9', linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    # a little headroom above the tallest bar
    ax.set_ylim(0, table.values.max() * 1.08)


def main():
    voc = load(VOC_CSV, BINS_VOC)
    brisc = load(BRISC_CSV, BINS_BRISC)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6),
                             gridspec_kw={'width_ratios': [5, 3]})
    draw(axes[0], voc, BINS_VOC, 'PASCAL VOC 2012')
    draw(axes[1], brisc, BINS_BRISC, 'BRISC 2025')

    # shared legend below the plots, so it never covers a bar
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
    labels = [lab for _k, lab in METHODS]
    fig.legend(handles, labels, loc='lower center', ncol=5,
               frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=9)

    fig.tight_layout(rect=[0, 0.09, 1, 1])
    fig.savefig(OUT_PDF, bbox_inches='tight')
    fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    print(f'written: {OUT_PDF} and {OUT_PNG}')

    # spot check against a few known values
    print('\n[spot check]')
    print(f"  VOC 0–5% ours(single): {voc.loc['SVD-Fusion (single)', '0–5%']:.2f} (expected ≈6.27)")
    print(f"  VOC 10–20% Eigen:      {voc.loc['EigenCAM', '10–20%']:.2f} (expected ≈2.02)")
    print(f"  BRISC 5–10% ours(single): {brisc.loc['SVD-Fusion (single)', '5–10%']:.2f} (expected ≈3.98)")


if __name__ == '__main__':
    main()
