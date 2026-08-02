"""Architecture overview figure for SVD-Fusion CAM.

Corrects three things relative to the earlier version:
  1. two taps at *different spatial resolutions*, not three consecutive layers
     (consecutive layers usually share a resolution, which is exactly the
     failure mode the paper argues against);
  2. the per-layer maps are drawn at different sizes, so the multi-resolution
     nature of the fusion is visible;
  3. two fusion weights instead of three, and the selection is annotated with
     the cell-coverage criterion that produces it.

Outputs Fig1_overview.pdf (for LaTeX) and Fig1_overview.svg (editable in
Inkscape, Illustrator or Affinity).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch, Circle

# ----------------------------------------------------------------- palette
SLAB_FACE = "#F2D5AC"      # conv slab front
SLAB_EDGE = "#C46A5A"
SLAB_SIDE = "#E0BE93"
CARD_FACE = "#FDF3D8"      # pipeline card
CARD_EDGE = "#C9A227"
STACK_COLS = ["#CBB9E0", "#F3C7CB", "#FBF1DC"]
BOX_GREEN = "#3F8F3F"
INK = "#1A1A1A"
GREY = "#7A7A7A"

W, H = 16.0, 8.2
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


# ----------------------------------------------------------------- helpers
def arrow(x0, y0, x1, y1, lw=1.4, color=INK, style="-|>", ms=9, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=ms, lw=lw, color=color,
                                 linestyle=ls, shrinkA=0, shrinkB=0,
                                 zorder=5))


def slab(x, y, h, w=0.42, depth=0.20, label="", res=""):
    """A 3-D looking convolutional slab; height encodes spatial resolution."""
    ax.add_patch(Rectangle((x + depth, y + depth), w, h, facecolor=SLAB_SIDE,
                           edgecolor=SLAB_EDGE, lw=1.0, zorder=2))
    ax.add_patch(Rectangle((x, y), w, h, facecolor=SLAB_FACE,
                           edgecolor=SLAB_EDGE, lw=1.2, zorder=3))
    if label:
        ax.text(x + w / 2, y + h / 2, label, rotation=90, ha="center",
                va="center", fontsize=8.5, fontweight="bold", color=INK,
                zorder=4)
    if res:
        ax.text(x + w / 2 + depth / 2, y - 0.30, res, ha="center", va="top",
                fontsize=8, color=GREY, zorder=8,
                bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                          edgecolor="none", alpha=0.95))
    return x + w + depth, y + h / 2          # right edge, mid height


def stack(x, y, w, h, n=3, dx=0.10, dy=0.10, label="", sub="", fs=8.5):
    """A small stack of cards, used for per-channel tensors."""
    for i in range(n - 1, -1, -1):
        ax.add_patch(Rectangle((x + i * dx, y + i * dy), w, h,
                               facecolor=STACK_COLS[i % len(STACK_COLS)],
                               edgecolor=CARD_EDGE, lw=0.9, zorder=3 + (n - i)))
    ax.add_patch(Rectangle((x, y), w, h, facecolor=CARD_FACE,
                           edgecolor=CARD_EDGE, lw=1.3, zorder=3 + n + 1))
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, fontstyle="italic", fontweight="bold",
                color=INK, zorder=3 + n + 2)
    if sub:
        ax.text(x + w / 2, y - 0.16, sub, ha="center", va="top",
                fontsize=9, color=INK, zorder=6)
    return x + w + (n - 1) * dx, y + h / 2


def matrix(x, y, w, h, ncol=5, label="", top=""):
    """A matrix drawn as vertical stripes (columns = channels)."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white",
                           edgecolor=INK, lw=1.2, zorder=3))
    cw = w / ncol
    for i in range(ncol):
        ax.add_patch(Rectangle((x + i * cw, y), cw, h,
                               facecolor=STACK_COLS[i % len(STACK_COLS)],
                               edgecolor="none", alpha=0.85, zorder=4))
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none",
                           edgecolor=INK, lw=1.2, zorder=5))
    if label:
        ax.text(x + w / 2, y - 0.26, label, ha="center", va="top",
                fontsize=10, fontweight="bold", zorder=6)
    if top:
        ax.text(x + w / 2, y + h + 0.10, top, ha="center", va="bottom",
                fontsize=8, color=GREY, zorder=6)
    return x + w, y + h / 2


def blob(cx, cy, size, grid, seed=0, cmap="jet"):
    """A synthetic saliency patch rendered at a given grid resolution."""
    rng = np.random.default_rng(seed)
    g = np.linspace(-1.6, 1.6, grid)
    xx, yy = np.meshgrid(g, g)
    z = np.exp(-((xx - 0.05) ** 2 + (yy + 0.05) ** 2) / 0.30)
    z += 0.12 * rng.random((grid, grid))
    z = (z - z.min()) / (z.max() - z.min())
    ax.imshow(z, cmap=cmap, extent=[cx - size / 2, cx + size / 2,
                                    cy - size / 2, cy + size / 2],
              zorder=4, interpolation="nearest")
    ax.add_patch(Rectangle((cx - size / 2, cy - size / 2), size, size,
                           facecolor="none", edgecolor=INK, lw=1.0, zorder=5))


# ========================================================== 1. backbone row
y_bb = 5.55                                   # baseline of the slabs
ax.text(0.15, 8.02, "Step 1  —  one forward and one backward pass",
        fontsize=10, fontweight="bold", color=INK)

# input image
rng = np.random.default_rng(3)
brain = np.clip(rng.normal(0.35, 0.16, (64, 64)), 0, 1)
yy, xx = np.mgrid[0:64, 0:64]
brain += 0.55 * np.exp(-(((xx - 32) ** 2 + (yy - 30) ** 2) / 380))
brain += 0.40 * np.exp(-(((xx - 40) ** 2 + (yy - 24) ** 2) / 60))
ax.imshow(np.clip(brain, 0, 1), cmap="gray",
          extent=[0.25, 1.45, y_bb + 0.05, y_bb + 1.25], zorder=3)
ax.add_patch(Rectangle((0.25, y_bb + 0.05), 1.2, 1.2, facecolor="none",
                       edgecolor=INK, lw=1.0, zorder=4))

# convolutional slabs: height encodes spatial resolution
slab_x = [2.05, 3.15, 4.25, 5.55, 6.85]
slab_h = [1.55, 1.30, 1.05, 0.80, 0.62]
slab_lab = ["Conv 1", "Conv 2", "Conv i", "Conv j", "Conv n"]
slab_res = ["", "", "", "", ""]
mids = []
for x, h, lab, res in zip(slab_x, slab_h, slab_lab, slab_res):
    r, m = slab(x, y_bb + (1.55 - h) / 2, h, label=lab, res=res)
    mids.append((x, r, m, h))

arrow(1.50, y_bb + 0.65, slab_x[0] - 0.04, y_bb + 0.65)
for i in range(len(slab_x) - 1):
    x0 = mids[i][1]
    x1 = slab_x[i + 1] - 0.04
    if i == 1:                                   # dotted gap: layers omitted
        arrow(x0, y_bb + 0.78, x1, y_bb + 0.78, ls=(0, (2.5, 2.5)))
    else:
        arrow(x0, y_bb + 0.78, x1, y_bb + 0.78)

# classifier head
fc_x = 7.75
ax.add_patch(Rectangle((fc_x + 0.16, y_bb + 0.62), 0.30, 0.55,
                       facecolor=SLAB_SIDE, edgecolor=SLAB_EDGE, lw=1.0, zorder=2))
ax.add_patch(Rectangle((fc_x, y_bb + 0.46), 0.30, 0.55, facecolor=SLAB_FACE,
                       edgecolor=SLAB_EDGE, lw=1.2, zorder=3))
ax.text(fc_x + 0.15, y_bb + 0.735, "FC", rotation=90, ha="center", va="center",
        fontsize=8.5, fontweight="bold", zorder=4)
arrow(mids[-1][1], y_bb + 0.78, fc_x - 0.04, y_bb + 0.78)
ax.text(fc_x + 0.78, y_bb + 0.26, r"class logit  $y^{c}$", fontsize=9.5,
        fontstyle="italic", fontweight="bold", ha="center", va="center")
ax.text(fc_x + 0.78, y_bb - 0.10, r"for target class $c$", fontsize=7.5,
        color=GREY, ha="center", va="center")
arrow(fc_x + 0.32, y_bb + 0.73, fc_x + 0.64, y_bb + 0.73, lw=1.1)

# --- the two selected levels -------------------------------------------
sel = [2, 4]                                      # mid-resolution and deepest
sel_labels = ["mid-resolution\n" + r"$30\times30$",
              "deepest\n" + r"$8\times8$"]
for idx, res_lab in zip(sel, sel_labels):
    x0, x1, _, h = mids[idx]
    cx = (x0 + x1) / 2
    ax.add_patch(Rectangle((x0 - 0.10, y_bb + (1.55 - h) / 2 - 0.10),
                           (x1 - x0) + 0.20, h + 0.20, facecolor="none",
                           edgecolor=BOX_GREEN, lw=1.6, linestyle="--",
                           zorder=6))
    top = y_bb + (1.55 - h) / 2 + h + 0.26
    ax.text(cx, top, res_lab, ha="center", va="bottom", fontsize=9.5,
            color=BOX_GREEN, fontweight="bold", zorder=8,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                      edgecolor=BOX_GREEN, lw=0.8, alpha=0.97))
    arrow(cx, y_bb + (1.55 - h) / 2 - 0.12, cx, 4.28, lw=1.5, color=BOX_GREEN)

ax.text((mids[2][0] + mids[4][1]) / 2, 7.62,
        "two levels at *different* resolutions, selected by cell coverage "
        r"$n_\ell=\rho H_\ell W_\ell$  (Sec. 3.4)",
        ha="center", va="bottom", fontsize=8.5, color=BOX_GREEN,
        fontstyle="italic")


# ==================================================== 2. per-level pipeline
box_x0, box_x1, box_y0, box_y1 = 0.20, 15.80, 0.45, 4.20
ax.add_patch(FancyBboxPatch((box_x0, box_y0), box_x1 - box_x0, box_y1 - box_y0,
                            boxstyle="round,pad=0.02,rounding_size=0.10",
                            facecolor="none", edgecolor=BOX_GREEN, lw=1.6,
                            linestyle=(0, (6, 4)), zorder=1))
ax.text(box_x0 + 0.18, box_y1 - 0.28,
        r"Step 2  —  applied independently to each selected level $\ell$",
        fontsize=10, fontweight="bold", color=BOX_GREEN, va="center")

yc = 2.10                                          # pipeline centre line
cw, ch = 1.35, 0.85                                # card size

x = 0.60
r1, _ = stack(x, yc - ch / 2, cw, ch, label="Feature\nMaps", sub=r"$A^{\ell}_{j}$")
arrow(r1 + 0.05, yc, r1 + 0.60, yc)

x = r1 + 0.65
r2, _ = stack(x, yc - ch / 2, cw, ch, label="Gradient\nMatrices",
              sub=r"$\partial y^{c}/\partial A^{\ell}_{j}$")
arrow(r2 + 0.05, yc, r2 + 0.55, yc)

x = r2 + 0.60
r3, _ = matrix(x, yc - 0.72, 0.95, 1.44, ncol=5, label=r"$\mathbf{G}$",
               top=r"$HW\times C$")
ax.text(r3 + 0.36, yc + 0.16, "SVD", ha="center", va="bottom", fontsize=8.5,
        fontstyle="italic")
arrow(r3 + 0.05, yc, r3 + 0.67, yc)

x = r3 + 0.72
r4, _ = matrix(x, yc - 0.72, 0.95, 1.44, ncol=5,
               label=r"$\widehat{\mathbf{G}}_k$", top="rank-$k$")
ax.text(r4 + 0.42, yc + 0.16, "reshape", ha="center", va="bottom",
        fontsize=8.5, fontstyle="italic")
arrow(r4 + 0.05, yc, r4 + 0.79, yc)

x = r4 + 0.84
r5, _ = stack(x, yc - ch / 2, cw, ch, label="Spatial\nWeight Maps",
              sub=r"$M^{\ell}_{j}$", fs=8)
arrow(r5 + 0.05, yc, r5 + 0.42, yc)

# Hadamard product
had_x = r5 + 0.62
ax.add_patch(Circle((had_x, yc), 0.20, facecolor="#D9D9D9", edgecolor=INK,
                    lw=1.2, zorder=6))
ax.text(had_x, yc, r"$\odot$", ha="center", va="center", fontsize=13, zorder=7)
arrow(had_x + 0.22, yc, had_x + 0.60, yc)

x = had_x + 0.65
r6, _ = stack(x, yc - ch / 2, cw, ch, label=r"$M^{\ell}_{j} \odot A^{\ell}_{j}$")
arrow(r6 + 0.05, yc, r6 + 0.42, yc)

# channel sum
sum_x = r6 + 0.62
ax.add_patch(Circle((sum_x, yc), 0.22, facecolor="#D9D9D9", edgecolor=INK,
                    lw=1.2, zorder=6))
ax.text(sum_x, yc, r"$\Sigma$", ha="center", va="center", fontsize=13, zorder=7)
ax.text(sum_x, yc - 0.42, "over channels", ha="center", va="top", fontsize=7.5,
        color=GREY)

# skip connection: feature maps feed the Hadamard product
skip_y = 0.70
ax.plot([0.60 + cw / 2, 0.60 + cw / 2], [yc - ch / 2 - 0.42, skip_y],
        color=INK, lw=1.2, zorder=4)
ax.plot([0.60 + cw / 2, had_x], [skip_y, skip_y], color=INK, lw=1.2, zorder=4)
arrow(had_x, skip_y, had_x, yc - 0.22, lw=1.2)
ax.text((0.60 + cw / 2 + had_x) / 2, skip_y - 0.12,
        "activations reused for the Hadamard product",
        ha="center", va="top", fontsize=7.5, color=GREY, fontstyle="italic")

# out of the box, up to the fusion stage
arrow(sum_x + 0.24, yc, 14.85, yc, lw=1.4)
ax.plot([14.85, 14.85], [yc, 3.72], color=INK, lw=1.4, zorder=4)
ax.plot([14.85, 11.55], [3.72, 3.72], color=INK, lw=1.4, zorder=4)
arrow(11.55, 3.72, 11.55, 4.32, lw=1.4)
ax.text(11.55, 3.58, "run once per selected level  "
        r"$\Rightarrow$  one saliency map each",
        ha="center", va="top", fontsize=8, color=GREY, fontstyle="italic")


# ============================================================== 3. fusion
ax.text(9.75, 8.02, "Step 3  —  resolution-aware fusion",
        fontsize=10, fontweight="bold", color=INK)

# the two per-level maps, drawn at *different* sizes
map_mid_c = (10.75, 5.25)
map_dep_c = (12.35, 5.25)
blob(*map_mid_c, size=1.30, grid=30, seed=1)      # fine grid
blob(*map_dep_c, size=0.85, grid=8, seed=2)       # coarse grid
ax.text(map_mid_c[0], map_mid_c[1] - 0.72, r"mid-res.  $30\times30$",
        ha="center", va="top", fontsize=8.5, zorder=8,
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                  edgecolor="none", alpha=0.95))
ax.text(map_dep_c[0], map_dep_c[1] - 0.49, r"deepest  $8\times8$",
        ha="center", va="top", fontsize=8.5, zorder=8,
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                  edgecolor="none", alpha=0.95))
ax.text(map_mid_c[0] - 0.16, map_mid_c[1] + 0.80,
        r"$\lambda_{\mathrm{mid}}$", ha="right", va="bottom", fontsize=10.5)
ax.text(map_dep_c[0] + 0.16, map_dep_c[1] + 0.58,
        r"$\lambda_{\mathrm{deep}}$", ha="left", va="bottom", fontsize=10.5)

# weighted sum
fuse_x, fuse_y = 11.55, 6.70
ax.add_patch(Circle((fuse_x, fuse_y), 0.22, facecolor="#D9D9D9",
                    edgecolor=INK, lw=1.2, zorder=6))
ax.text(fuse_x, fuse_y, r"$\Sigma$", ha="center", va="center", fontsize=13,
        zorder=7)
ax.plot([map_mid_c[0], map_mid_c[0]], [map_mid_c[1] + 0.95, fuse_y],
        color=INK, lw=1.2, zorder=4)
ax.plot([map_mid_c[0], fuse_x - 0.22], [fuse_y, fuse_y], color=INK, lw=1.2,
        zorder=4)
ax.plot([map_dep_c[0], map_dep_c[0]], [map_dep_c[1] + 0.72, fuse_y],
        color=INK, lw=1.2, zorder=4)
ax.plot([map_dep_c[0], fuse_x + 0.22], [fuse_y, fuse_y], color=INK, lw=1.2,
        zorder=4)
ax.text(fuse_x, fuse_y + 0.34,
        r"$\lambda_{\mathrm{mid}}+\lambda_{\mathrm{deep}}=1$",
        ha="center", va="bottom", fontsize=8.5, color=GREY, zorder=8,
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                  edgecolor="none", alpha=0.95))

# upscale to the final explanation
arrow(fuse_x + 0.24, fuse_y, 13.45, fuse_y, lw=1.3)
ax.text(12.80, fuse_y + 0.17, "upscale", ha="center", va="bottom",
        fontsize=8.5, fontstyle="italic")
ax.imshow(np.clip(brain, 0, 1), cmap="gray",
          extent=[13.65, 14.95, fuse_y - 0.65, fuse_y + 0.65], zorder=3)
heat = np.exp(-(((xx - 40) ** 2 + (yy - 24) ** 2) / 130))
ax.imshow(heat, cmap="jet", alpha=0.55,
          extent=[13.65, 14.95, fuse_y - 0.65, fuse_y + 0.65], zorder=4)
ax.add_patch(Rectangle((13.65, fuse_y - 0.65), 1.30, 1.30, facecolor="none",
                       edgecolor=INK, lw=1.0, zorder=5))
ax.text(14.30, fuse_y - 0.80, "explanation", ha="center", va="top",
        fontsize=8.5)

fig.savefig("Fig1_overview.pdf", bbox_inches="tight")
fig.savefig("Fig1_overview.svg", bbox_inches="tight")
fig.savefig("Fig1_overview.png", dpi=200, bbox_inches="tight")
print("wrote Fig1_overview.pdf, .svg and .png")
