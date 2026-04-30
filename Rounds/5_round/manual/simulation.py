# -*- coding: utf-8 -*-
"""
Round 5 Manual: PAIRWISE PARAMETER SIMULATION
==============================================

Comprehensive sweep across all C(9,2)=36 product pairs.

For each pair (i, j):
  - Vary r_i and r_j over [0%, 60%] on a GRID x GRID mesh
  - Hold all other 7 products at their base-case expected returns
  - Compute the total optimal portfolio PnL at every grid point
  - Plot as a 2D heatmap with contour lines

Output
------
  fig1_pairwise_simulation.png  -- 9x9 matrix: heatmaps (lower triangle)
                                   + individual sensitivity curves (diagonal)
  fig2_optimisation_summary.png -- sensitivity ranking, top pairs, top-3 zoomed heatmaps
"""

import sys
import io
import warnings
import numpy as np

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")

import os
import pathlib

# Save figures alongside this script, regardless of cwd
SCRIPT_DIR = pathlib.Path(__file__).parent

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.cm import ScalarMappable

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

BUDGET    = 1_000_000
GRID      = 50      # grid points per axis for heatmaps
DIAG_PTS  = 120     # points for individual sensitivity curves
R_MAX     = 0.60    # x/y axis cap: 60% expected return

PRODUCTS = [
    # name,               short, direction, r_base, colour
    ("Lava Cake",         "LC",  "SELL",    0.40,   "#e74c3c"),
    ("Obsidian Cutlery",  "OC",  "BUY",     0.30,   "#e67e22"),
    ("Pyroflex Cells",    "PC",  "SELL",    0.25,   "#f39c12"),
    ("Magma Ink",         "MI",  "BUY",     0.25,   "#9b59b6"),
    ("Sulfur Reactor",    "SR",  "BUY",     0.20,   "#3498db"),
    ("Thermalite Core",   "TC",  "BUY",     0.20,   "#1abc9c"),
    ("Scoria Paste",      "SP",  "BUY",     0.15,   "#27ae60"),
    ("Volcanic Incense",  "VI",  "BUY",     0.15,   "#16a085"),
    ("Ashes of Phoenix",  "AP",  "SELL",    0.10,   "#95a5a6"),
]
N       = len(PRODUCTS)
NAMES   = [p[0] for p in PRODUCTS]
SHORTS  = [p[1] for p in PRODUCTS]
DIRS    = [p[2] for p in PRODUCTS]
BASE_R  = np.array([p[3] for p in PRODUCTS])
COLOURS = [p[4] for p in PRODUCTS]

# ============================================================================
# 2. PORTFOLIO MODEL
# ============================================================================

def portfolio_pnl(returns: np.ndarray, budget: float = BUDGET) -> float:
    """
    Given a vector of expected return magnitudes, compute total net PnL
    at each product's unconstrained optimal allocation (p* = 50*r),
    then scale proportionally if the sum exceeds 100%.
    """
    raw    = 50.0 * np.asarray(returns, dtype=float)
    total  = raw.sum()
    scale  = min(1.0, 100.0 / total) if total > 0 else 1.0
    allocs = raw * scale
    frac   = allocs / 100.0
    return float((frac * budget * returns - frac**2 * budget).sum())


BASE_PNL = portfolio_pnl(BASE_R)
print(f"Base-case portfolio PnL : {BASE_PNL:,.0f} XIRECs")

# ============================================================================
# 3. DIAGONAL: individual sensitivity curves
# ============================================================================

print("Computing individual sensitivity curves...")
X_DIAG   = np.linspace(0.0, R_MAX, DIAG_PTS)
diag_y   = np.zeros((N, DIAG_PTS))

for i in range(N):
    for k, xi in enumerate(X_DIAG):
        r = BASE_R.copy(); r[i] = xi
        diag_y[i, k] = portfolio_pnl(r)

# ============================================================================
# 4. PAIRWISE GRIDS (lower triangle: row > col)
# ============================================================================

print("Computing 36 pairwise grids (GRID=%d x %d)..." % (GRID, GRID))
X_GRID = np.linspace(0.0, R_MAX, GRID)   # same axis range for all products

grids = {}   # (i, j) -> Z  where i > j
for i in range(N):
    for j in range(i):
        Z = np.zeros((GRID, GRID))
        for ii in range(GRID):
            for jj in range(GRID):
                r = BASE_R.copy()
                r[i] = X_GRID[ii]
                r[j] = X_GRID[jj]
                Z[ii, jj] = portfolio_pnl(r)
        grids[(i, j)] = Z
        print(f"  [{i},{j}]  {SHORTS[i]} x {SHORTS[j]}", end="\r")

print("\nAll grids computed.                    ")

# ============================================================================
# 5. SENSITIVITY: central-difference |dPnL/dr| per product
# ============================================================================

DELTA = 0.005
sens = np.zeros(N)
for i in range(N):
    rp = BASE_R.copy(); rp[i] += DELTA
    rm = BASE_R.copy(); rm[i] -= DELTA
    sens[i] = abs(portfolio_pnl(rp) - portfolio_pnl(rm)) / (2 * DELTA)

sens_order = np.argsort(sens)[::-1]

# Pair impact: PnL range across full grid
pair_impact = []
for (i, j), Z in grids.items():
    pair_impact.append((Z.max() - Z.min(), i, j))
pair_impact.sort(reverse=True)

# ============================================================================
# 6. COLOUR MAP & NORMALISATION
# ============================================================================

cmap_rg = LinearSegmentedColormap.from_list(
    "rg",
    ["#6B0000", "#CC2222", "#FF8888",
     "#FFFFFF",
     "#88FF88", "#22CC22", "#006B00"],
)

all_Z = [grids[(i, j)] for i in range(N) for j in range(i)]
g_min = min(z.min() for z in all_Z)
g_max = max(z.max() for z in all_Z)

# Guard against BASE_PNL being exactly at an edge
vc = float(np.clip(BASE_PNL, g_min + 1, g_max - 1))
norm_global = TwoSlopeNorm(vmin=g_min, vcenter=vc, vmax=g_max)

# ============================================================================
# 7. FIGURE 1 — 9 x 9 PAIRWISE MATRIX
# ============================================================================

print("Generating Figure 1: 9x9 pairwise matrix...")

FIG1_W, FIG1_H = 30, 28
fig1 = plt.figure(figsize=(FIG1_W, FIG1_H), facecolor="#0a0a0a")

outer = gridspec.GridSpec(
    N, N, figure=fig1,
    left=0.07, right=0.93, top=0.93, bottom=0.05,
    hspace=0.08, wspace=0.08,
)

for row in range(N):
    for col in range(N):
        ax = fig1.add_subplot(outer[row, col])
        ax.set_facecolor("#111111")
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a2a2a"); sp.set_linewidth(0.4)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # ── Diagonal: individual sensitivity curve ──────────────────────────
        if row == col:
            y = diag_y[row]
            ax.fill_between(X_DIAG * 100, y, 0,
                            where=y >= 0, color=COLOURS[row], alpha=0.25, zorder=1)
            ax.fill_between(X_DIAG * 100, y, 0,
                            where=y < 0,  color="#e74c3c",    alpha=0.25, zorder=1)
            ax.plot(X_DIAG * 100, y, color=COLOURS[row], lw=1.6, zorder=3)
            ax.axhline(0,                  color="#444444", lw=0.5, ls="--")
            ax.axhline(BASE_PNL,           color="#777777", lw=0.4, ls=":")
            ax.axvline(BASE_R[row] * 100,  color="white",   lw=1.1, ls="--", alpha=0.8)
            ax.set_xlim(0, R_MAX * 100)
            ypad = max(abs(y.min()), abs(y.max())) * 0.08
            ax.set_ylim(y.min() - ypad, y.max() + ypad)
            sign = "+" if DIRS[row] == "BUY" else "\u2212"
            ax.text(0.50, 0.96, SHORTS[row],
                    transform=ax.transAxes, ha="center", va="top",
                    color="white", fontsize=7.5, fontweight="bold")
            ax.text(0.50, 0.82, f"{sign}{BASE_R[row]*100:.0f}%",
                    transform=ax.transAxes, ha="center", va="top",
                    color=COLOURS[row], fontsize=6.5)

        # ── Lower triangle: pairwise heatmap ────────────────────────────────
        elif row > col:
            Z = grids[(row, col)]
            ax.pcolormesh(X_GRID * 100, X_GRID * 100, Z,
                          cmap=cmap_rg, norm=norm_global,
                          shading="auto", rasterized=True)
            # Contours at base-PnL and zero
            for level, lc, lw, ls in [
                (0,        "#FF4444", 1.0, "--"),
                (BASE_PNL, "white",   0.8, "-"),
            ]:
                try:
                    ax.contour(X_GRID * 100, X_GRID * 100, Z,
                               levels=[level], colors=[lc],
                               linewidths=[lw], linestyles=[ls], alpha=0.8)
                except Exception:
                    pass
            # Base-case star
            ax.plot(BASE_R[col] * 100, BASE_R[row] * 100,
                    "*", color="white", ms=5, zorder=5)
            ax.set_xlim(0, R_MAX * 100)
            ax.set_ylim(0, R_MAX * 100)

        # ── Upper triangle: blank ────────────────────────────────────────────
        else:
            ax.set_facecolor("#0a0a0a")
            ax.axis("off")
            # Put product labels along the diagonal of the upper triangle
            if row == 0 and col < N:
                ax.text(0.5, 0.5, SHORTS[col],
                        transform=ax.transAxes, ha="center", va="center",
                        color=COLOURS[col], fontsize=7.5, fontweight="bold")

        # Row/column outer labels
        if col == 0 and row > 0:
            ax.set_ylabel(SHORTS[row], color=COLOURS[row], fontsize=7,
                          fontweight="bold", rotation=0, labelpad=14, va="center")
        if row == N - 1 and col > 0:
            ax.set_xlabel(SHORTS[col], color=COLOURS[col], fontsize=7,
                          fontweight="bold", labelpad=6)

# Shared colour bar
sm = ScalarMappable(cmap=cmap_rg, norm=norm_global)
sm.set_array([])
cb_ax = fig1.add_axes([0.94, 0.08, 0.012, 0.82])
cb = fig1.colorbar(sm, cax=cb_ax)
cb.set_label("Portfolio PnL (XIRECs)", color="white", fontsize=9, labelpad=8)
cb.ax.yaxis.set_tick_params(color="white", labelsize=7)
plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white", fontsize=7)

# Legend box (upper-right area)
leg_ax = fig1.add_axes([0.93, 0.93, 0.06, 0.04])
leg_ax.axis("off")

fig1.suptitle(
    "IGNITH EXCHANGE — PAIRWISE PARAMETER SIMULATION\n"
    "Lower triangle: total portfolio PnL as both products' expected returns vary "
    "(others fixed at base).   "
    "Diagonal: single-product sensitivity.   "
    f"White \u2605 = base case.   "
    f"White contour = base PnL ({BASE_PNL:,.0f}).   "
    "Red dashed = break-even.",
    color="white", fontsize=9, y=0.975,
)

out1 = SCRIPT_DIR / "fig1_pairwise_simulation.png"
fig1.savefig(out1, dpi=120, bbox_inches="tight", facecolor="#0a0a0a")
plt.close(fig1)
print(f"Saved {out1}")

# ============================================================================
# 8. FIGURE 2 — OPTIMISATION SUMMARY
# ============================================================================

print("Generating Figure 2: optimisation summary...")

fig2 = plt.figure(figsize=(22, 14), facecolor="#0a0a0a")
gs2 = gridspec.GridSpec(2, 3, figure=fig2,
                         left=0.07, right=0.97, top=0.90, bottom=0.07,
                         hspace=0.45, wspace=0.38)

dark_kw = dict(facecolor="#111111")
tick_kw = dict(colors="white", labelsize=8)
spine_c = "#333333"

def style_ax(ax):
    ax.set_facecolor("#111111")
    ax.tick_params(**tick_kw)
    for sp in ax.spines.values():
        sp.set_edgecolor(spine_c)

# ── Panel A: sensitivity ranking ────────────────────────────────────────────
ax_a = fig2.add_subplot(gs2[0, 0])
style_ax(ax_a)
sorted_idx  = sens_order
sorted_sens = [sens[i] for i in sorted_idx]
sorted_name = [NAMES[i]  for i in sorted_idx]
sorted_col  = [COLOURS[i] for i in sorted_idx]
bars_a = ax_a.barh(sorted_name, sorted_sens,
                   color=sorted_col, edgecolor="#1a1a1a", linewidth=0.5)
ax_a.set_xlabel("|dPnL/dr|  (XIRECs per unit return)", color="white", fontsize=8)
ax_a.set_title("Product Sensitivity Ranking\n"
               "Effect on total PnL per unit change in return estimate",
               color="white", fontsize=8.5, fontweight="bold", pad=6)
ax_a.invert_yaxis()
ax_a.tick_params(axis="y", labelsize=7.5)
for bar, s in zip(bars_a, sorted_sens):
    ax_a.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
              f"{s:,.0f}", va="center", color="white", fontsize=6.5)

# ── Panel B: all sensitivity curves overlaid ────────────────────────────────
ax_b = fig2.add_subplot(gs2[0, 1])
style_ax(ax_b)
for i in range(N):
    ax_b.plot(X_DIAG * 100, diag_y[i] / 1000,
              color=COLOURS[i], lw=1.6, label=SHORTS[i], alpha=0.9)
    ax_b.axvline(BASE_R[i] * 100, color=COLOURS[i], lw=0.5, ls=":", alpha=0.4)
ax_b.axhline(0,              color="#555555", lw=0.8, ls="--")
ax_b.axhline(BASE_PNL / 1000, color="#ffffff", lw=0.7, ls="--", alpha=0.35)
ax_b.set_xlabel("Expected Return (%)", color="white", fontsize=8.5)
ax_b.set_ylabel("Portfolio PnL  (k XIRECs)", color="white", fontsize=8.5)
ax_b.set_title("Individual Sensitivity Curves\n"
               "Total PnL when each product's return varies (others at base)",
               color="white", fontsize=8.5, fontweight="bold", pad=6)
ax_b.legend(fontsize=6.5, ncol=3, loc="upper left",
            facecolor="#1e1e1e", edgecolor="#444444", labelcolor="white")
ax_b.set_xlim(0, R_MAX * 100)

# ── Panel C: top-12 pair impact bar chart ───────────────────────────────────
ax_c = fig2.add_subplot(gs2[0, 2])
style_ax(ax_c)
top12      = pair_impact[:12]
pair_names = [f"{SHORTS[i]}\u00d7{SHORTS[j]}" for _, i, j in top12]
pair_vals  = [v for v, _, _ in top12]
pair_cols  = [COLOURS[i] for _, i, j in top12]
bars_c = ax_c.barh(pair_names, pair_vals, color=pair_cols,
                   edgecolor="#1a1a1a", linewidth=0.5)
ax_c.set_xlabel("PnL range across full parameter sweep (XIRECs)", color="white", fontsize=8)
ax_c.set_title("Top 12 Most Impactful Product Pairs\n"
               "Ranked by PnL variance across joint return sweep",
               color="white", fontsize=8.5, fontweight="bold", pad=6)
ax_c.invert_yaxis()
ax_c.tick_params(axis="y", labelsize=7.5)

# ── Panels D/E/F: top-3 pairwise heatmaps (zoomed, with contour labels) ─────
for k in range(3):
    impact, i, j = pair_impact[k]
    ax_k = fig2.add_subplot(gs2[1, k])
    style_ax(ax_k)
    Z = grids[(i, j)]

    v_dev  = max(abs(Z.min() - BASE_PNL), abs(Z.max() - BASE_PNL))
    v_low  = BASE_PNL - v_dev
    v_high = BASE_PNL + v_dev
    vc_k   = float(np.clip(BASE_PNL, v_low + 1, v_high - 1))
    try:
        norm_k = TwoSlopeNorm(vmin=v_low, vcenter=vc_k, vmax=v_high)
    except Exception:
        from matplotlib.colors import Normalize
        norm_k = Normalize(vmin=v_low, vmax=v_high)

    im = ax_k.pcolormesh(X_GRID * 100, X_GRID * 100, Z,
                         cmap=cmap_rg, norm=norm_k,
                         shading="auto", rasterized=True)

    # Labelled contour lines
    n_levels = 8
    levels = np.linspace(Z.min(), Z.max(), n_levels)
    try:
        cs = ax_k.contour(X_GRID * 100, X_GRID * 100, Z,
                          levels=levels, colors="white",
                          linewidths=0.5, alpha=0.55)
        ax_k.clabel(cs, inline=True, fontsize=5.5,
                    fmt=lambda v: f"{v/1000:.0f}k", colors="white",
                    inline_spacing=1)
    except Exception:
        pass

    # Zero PnL contour in red
    try:
        ax_k.contour(X_GRID * 100, X_GRID * 100, Z, levels=[0],
                     colors=["#FF3333"], linewidths=[1.5], linestyles=["--"])
    except Exception:
        pass

    # Base-case marker
    ax_k.plot(BASE_R[j] * 100, BASE_R[i] * 100, "*",
              color="white", ms=12, zorder=6,
              markeredgecolor="#000000", markeredgewidth=0.5)

    ax_k.set_xlim(0, R_MAX * 100)
    ax_k.set_ylim(0, R_MAX * 100)
    ax_k.set_xlabel(f"{NAMES[j]} return (%)", color=COLOURS[j], fontsize=8.5)
    ax_k.set_ylabel(f"{NAMES[i]} return (%)", color=COLOURS[i], fontsize=8.5)
    ax_k.set_title(
        f"#{k+1} Highest Impact: {SHORTS[i]} \u00d7 {SHORTS[j]}\n"
        f"PnL range = {impact:,.0f} XIRECs",
        color="white", fontsize=8.5, fontweight="bold", pad=6,
    )

    cb2 = fig2.colorbar(im, ax=ax_k, fraction=0.046, pad=0.03)
    cb2.ax.yaxis.set_tick_params(color="white", labelsize=6.5)
    plt.setp(plt.getp(cb2.ax.axes, "yticklabels"), color="white", fontsize=6.5)

fig2.suptitle(
    "IGNITH EXCHANGE — OPTIMISATION SUMMARY\n"
    f"Base portfolio: {BASE_PNL:,.0f} XIRECs   |   "
    f"Most sensitive product: {NAMES[sens_order[0]]}  "
    f"(sens = {sens[sens_order[0]]:,.0f} XIRECs/unit)   |   "
    f"Most impactful pair: {NAMES[pair_impact[0][1]]} \u00d7 {NAMES[pair_impact[0][2]]}",
    color="white", fontsize=10, fontweight="bold", y=0.965,
)

out2 = SCRIPT_DIR / "fig2_optimisation_summary.png"
fig2.savefig(out2, dpi=120, bbox_inches="tight", facecolor="#0a0a0a")
plt.close(fig2)
print(f"Saved {out2}")

# ============================================================================
# 9. CONSOLE OPTIMISATION REPORT
# ============================================================================

SEP = "=" * 72

print(f"\n{SEP}")
print("  OPTIMISATION FINDINGS")
print(SEP)

print(f"\n  Base-case total PnL : {BASE_PNL:>12,.0f} XIRECs")

print("\n  Product sensitivity ranking  (|dPnL/dr| at base case):")
for rank, i in enumerate(sens_order):
    bar = "#" * int(sens[i] / 5000)
    print(f"    {rank+1}. {NAMES[i]:<22}  {sens[i]:>10,.0f}  {bar}")

print("\n  Top 10 most impactful product pairs  (PnL range over full sweep):")
for rank, (rng, i, j) in enumerate(pair_impact[:10]):
    print(f"    {rank+1:2}. {NAMES[i]:<22}  x  {NAMES[j]:<22}  range = {rng:>10,.0f}")

print("\n  Maximum achievable PnL  (all returns at 2x base, capped at 60%):")
r_2x    = np.minimum(BASE_R * 2, 0.60)
pnl_2x  = portfolio_pnl(r_2x)
print(f"    {pnl_2x:>12,.0f} XIRECs  (+{(pnl_2x/BASE_PNL - 1)*100:.0f}% vs base)")

print("\n  Minimum PnL  (all returns drop to half base):")
r_half  = BASE_R * 0.5
pnl_half = portfolio_pnl(r_half)
print(f"    {pnl_half:>12,.0f} XIRECs  ({(pnl_half/BASE_PNL - 1)*100:.0f}% vs base)")

print("\n  Return threshold at which each product becomes net-positive:")
for i in range(N):
    # p* = 50r, net = 250,000 * r^2 > 0 for any r > 0 => always positive
    # But with budget constraint, a product becomes net negative only if
    # its return is 0 (zero allocation = zero PnL).
    # Show the return at which the individual contribution equals its fee.
    # net_i = 250,000 * r^2 > 0 for all r > 0 — so always positive when r > 0
    print(f"    {NAMES[i]:<22}  always positive for r > 0%  "
          f"(optimal alloc = {50*BASE_R[i]:.1f}%  |  "
          f"net = {250_000*BASE_R[i]**2:,.0f} XIRECs at base)")

print(f"\n{SEP}")
print("  Figures saved:")
print(f"    {out1}")
print(f"    {out2}")
print(SEP)
