"""
Round 4 Manual Challenge: Visual analysis and risk/hedging study.
Run: python analysis.py
Produces four figures saved as PNG files in the same folder.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # save to file; swap to "TkAgg" for interactive
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm

# ============================================================
# Simulation (same params as simulate.py)
# ============================================================
S0 = 50.0
SIGMA = 2.51
R = 0.0
TRADING_DAYS_PER_YEAR = 252
STEPS_PER_DAY = 4
STEPS_PER_YEAR = TRADING_DAYS_PER_YEAR * STEPS_PER_DAY
DT = 1.0 / STEPS_PER_YEAR

T2_STEPS = 2 * 5 * STEPS_PER_DAY   # 40
T3_STEPS = 3 * 5 * STEPS_PER_DAY   # 60
T2_YEARS = (2 * 5) / TRADING_DAYS_PER_YEAR
T3_YEARS = (3 * 5) / TRADING_DAYS_PER_YEAR

# Confirmed from competition rules
BINARY_PAYOFF = 10.0   # AC_40_BP: fixed cash payoff if S_T < 40 at expiry
KO_BARRIER    = 35.0   # AC_45_KO: knocked out if price ever falls below 35
CONTRACT_SIZE = 3000

N_SIM  = 300_000
N_PATH = 200_000   # separate smaller batch for path plots (same seed)
SEED = 42

print("Simulating paths...")
rng = np.random.default_rng(SEED)
Z = rng.standard_normal((N_SIM, T3_STEPS))
log_inc = (R - 0.5 * SIGMA**2) * DT + SIGMA * np.sqrt(DT) * Z
prices  = S0 * np.exp(np.cumsum(log_inc, axis=1))   # (N_SIM, 60)

S_2wk = prices[:, T2_STEPS - 1]
S_3wk = prices[:, T3_STEPS - 1]

# barrier breach = ANY step 1..59 below KO_BARRIER
barrier_breach = np.any(prices[:, :T3_STEPS - 1] < KO_BARRIER, axis=1)

# ── Payoffs ──────────────────────────────────────────────────────────────────
p_P2  = np.maximum(50 - S_2wk, 0)
p_C2  = np.maximum(S_2wk - 50, 0)
p_CO  = np.where(S_2wk >= 50, np.maximum(S_3wk - 50, 0), np.maximum(50 - S_3wk, 0))
p_BP  = np.where(S_3wk < 40, BINARY_PAYOFF, 0.0)
p_KO  = np.where(barrier_breach, 0.0, np.maximum(45 - S_3wk, 0))
p_45P = np.maximum(45 - S_3wk, 0)   # vanilla AC_45_P (for hedge)
p_35P = np.maximum(35 - S_3wk, 0)   # vanilla AC_35_P (for hedge)

# ── Per-path PnL for each recommended trade (× contract size already) ────────
# BUY trades: payoff - entry cost
# SELL trades: entry price received - payoff owed
pnl_P2 = 50  * (p_P2  - 9.75)   * CONTRACT_SIZE   # BUY 50 at ask 9.75
pnl_C2 = 50  * (p_C2  - 9.75)   * CONTRACT_SIZE   # BUY 50 at ask 9.75
pnl_CO = 50  * (22.20 - p_CO)   * CONTRACT_SIZE   # SELL 50 at bid 22.20
pnl_BP = 50  * (5.00  - p_BP)   * CONTRACT_SIZE   # SELL 50 at bid 5.00
pnl_KO = 500 * (p_KO  - 0.175) * CONTRACT_SIZE   # BUY 500 at ask 0.175 (barrier=35, fair=0.252)

# Portfolio variants
pnl_base    = pnl_P2 + pnl_C2 + pnl_CO + pnl_BP          # no KO
pnl_full    = pnl_base + pnl_KO                             # all 5 trades
# Hedge A: sell 500 AC_45_P at bid (9.05) — offsets KO's vanilla-put exposure, crystalises edge early
pnl_hedgeA  = pnl_full + 500 * (9.05 - p_45P) * CONTRACT_SIZE
# Hedge B: buy 500 AC_35_P at ask (4.35) — cheap barrier insurance: pays off when KO is knocked out
pnl_hedgeB  = pnl_full + 500 * (p_35P - 4.35) * CONTRACT_SIZE

# ── Helper: risk statistics ───────────────────────────────────────────────────
def risk_stats(pnl, label=""):
    e   = pnl.mean()
    sd  = pnl.std()
    var = np.percentile(pnl, 5)
    cvar = pnl[pnl <= var].mean()
    p_loss = (pnl < 0).mean()
    return {"label": label, "E[PnL]": e, "Std": sd,
            "VaR 5%": var, "CVaR 5%": cvar, "P(loss)": p_loss}

# ── Styling ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor":   "#16213e",
    "axes.edgecolor":   "#4a4a6a",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#e0e0e0",
    "ytick.color":      "#e0e0e0",
    "text.color":       "#e0e0e0",
    "grid.color":       "#2a2a4a",
    "grid.alpha":       0.5,
    "axes.titlecolor":  "#ffffff",
    "figure.titlesize": 13,
    "axes.titlesize":   11,
    "axes.labelsize":   9,
    "font.family":      "monospace",
})
CYAN   = "#00d4ff"
ORANGE = "#ff6b35"
GREEN  = "#2ecc71"
RED    = "#e74c3c"
PURPLE = "#9b59b6"
YELLOW = "#f1c40f"
GREY   = "#7f8c8d"

def vline(ax, x, label, color, lw=1.2):
    ax.axvline(x, color=color, lw=lw, ls="--", alpha=0.8)
    ax.text(x, ax.get_ylim()[1] * 0.92, f" {label}", color=color,
            fontsize=7, rotation=90, va="top")

# ============================================================
# FIGURE 1 — GBM paths + price distribution
# ============================================================
print("Plotting Figure 1: GBM paths + price distribution...")
fig1, axes = plt.subplots(1, 2, figsize=(14, 6))
fig1.suptitle("Figure 1  |  AC Price Dynamics  (S0=50, vol=251%, 3 weeks)")

# ── Left: sample paths ────────────────────────────────────────────────────────
ax = axes[0]
ax.set_title("100 sample GBM paths (coloured by final price)")
steps = np.arange(1, T3_STEPS + 1)
day_ticks = np.arange(0, T3_STEPS + 1, STEPS_PER_DAY)
sample_idx = np.random.default_rng(99).choice(N_SIM, 100, replace=False)
for i in sample_idx:
    s_final = prices[i, -1]
    if s_final > 70:
        c = GREEN
    elif s_final < 30:
        c = RED
    else:
        c = CYAN
    ax.plot(steps, prices[i], color=c, lw=0.4, alpha=0.45)

ax.axhline(S0,         color=YELLOW, lw=1.2, ls="-",  alpha=0.8, label=f"S0={S0}")
ax.axhline(KO_BARRIER, color=RED,    lw=1.5, ls="--", alpha=0.9, label=f"KO barrier={KO_BARRIER}")
ax.axhline(45,         color=ORANGE, lw=1.0, ls=":",  alpha=0.7, label="K=45 (KO strike)")
ax.axvline(T2_STEPS,   color=PURPLE, lw=1.0, ls="--", alpha=0.7, label="2wk (chooser choice)")

ax.set_xlabel("Step (4 per trading day)")
ax.set_ylabel("AC price")
ax.legend(fontsize=7, loc="upper right")
ax.set_xticks(day_ticks)
ax.set_xticklabels([f"D{i//STEPS_PER_DAY}" for i in day_ticks], fontsize=7)
ax.grid(True, lw=0.4)
patch_up   = mpatches.Patch(color=GREEN, label=">70")
patch_mid  = mpatches.Patch(color=CYAN,  label="30-70")
patch_down = mpatches.Patch(color=RED,   label="<30")
ax.legend(handles=[patch_up, patch_mid, patch_down], fontsize=7, loc="upper left")

# ── Right: price distribution at expiry ──────────────────────────────────────
ax = axes[1]
ax.set_title("Distribution of AC price at expiry (3wk, clipped at [0, 200])")
clip_prices = np.clip(S_3wk, 0, 200)
ax.hist(clip_prices, bins=200, color=CYAN, alpha=0.7, density=True, label="S_3wk")
ax.hist(np.clip(S_2wk, 0, 200), bins=200, color=PURPLE, alpha=0.4, density=True, label="S_2wk")

for k, col, lbl in [(35, RED, "K=35"), (40, ORANGE, "K=40"),
                    (45, YELLOW, "K=45"), (50, GREEN, "K=50"), (60, CYAN, "K=60")]:
    ax.axvline(k, color=col, lw=1.2, ls="--", alpha=0.85)
    ax.text(k + 0.5, ax.get_ylim()[1] * 0.5, lbl, color=col, fontsize=7, rotation=90)

pct_below_45 = (S_3wk < 45).mean() * 100
ax.text(0.02, 0.95, f"P(S_3wk < 45) = {pct_below_45:.1f}%\nP(S_3wk < 40) = {(S_3wk<40).mean()*100:.1f}%",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round", fc="#16213e", ec=CYAN, alpha=0.8))
ax.set_xlabel("AC price at expiry")
ax.set_ylabel("Density")
ax.legend(fontsize=8)
ax.grid(True, lw=0.4)

plt.tight_layout()
fig1.savefig("fig1_paths_distribution.png", dpi=150, bbox_inches="tight")
print("  Saved fig1_paths_distribution.png")

# ============================================================
# FIGURE 2 — KO option deep dive
# ============================================================
print("Plotting Figure 2: KO option analysis...")
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle("Figure 2  |  AC 45 KO Deep Dive  (barrier=45, K=45)")

survived = ~barrier_breach
itm_survived = survived & (S_3wk < 45)
otm_survived = survived & (S_3wk >= 45)
itm_breach   = barrier_breach & (S_3wk < 45)
otm_breach   = barrier_breach & (S_3wk >= 45)

# ── Top-left: 80 paths coloured by KO status ─────────────────────────────────
ax = axes[0, 0]
ax.set_title("80 sample paths — red=knocked out, green=survived ITM, blue=survived OTM")
rng2 = np.random.default_rng(7)
for mask, col, n in [(barrier_breach, RED, 60), (itm_survived, GREEN, 12), (otm_survived, CYAN, 8)]:
    idx_pool = np.where(mask)[0]
    if len(idx_pool) == 0:
        continue
    chosen = rng2.choice(idx_pool, min(n, len(idx_pool)), replace=False)
    for i in chosen:
        ax.plot(steps, prices[i], color=col, lw=0.5, alpha=0.55)
ax.axhline(KO_BARRIER, color=RED,    lw=2.0, ls="--", alpha=0.95, label=f"Barrier={KO_BARRIER}")
ax.axhline(45,         color=ORANGE, lw=1.2, ls=":",  alpha=0.80, label="Strike=45")
ax.set_xlabel("Step"); ax.set_ylabel("AC price")
red_p  = mpatches.Patch(color=RED,   label=f"Knocked out ({barrier_breach.mean()*100:.0f}%)")
grn_p  = mpatches.Patch(color=GREEN, label=f"Survived ITM ({itm_survived.mean()*100:.1f}%)")
cyn_p  = mpatches.Patch(color=CYAN,  label=f"Survived OTM ({otm_survived.mean()*100:.1f}%)")
ax.legend(handles=[red_p, grn_p, cyn_p], fontsize=7)
ax.grid(True, lw=0.4)

# ── Top-right: pie breakdown ──────────────────────────────────────────────────
ax = axes[0, 1]
ax.set_title("Scenario breakdown (all simulated paths)")
counts = [barrier_breach.sum(), itm_survived.sum(), otm_survived.sum()]
labels = [
    f"Knocked out\n({barrier_breach.mean()*100:.1f}%)",
    f"Survived ITM\n({itm_survived.mean()*100:.2f}%)",
    f"Survived OTM\n({otm_survived.mean()*100:.2f}%)",
]
colors = [RED, GREEN, CYAN]
wedges, texts, autotexts = ax.pie(counts, labels=labels, colors=colors,
                                   autopct="%1.1f%%", startangle=90,
                                   textprops={"fontsize": 8})
for at in autotexts:
    at.set_color("white")

# ── Bottom-left: KO payoff distribution (non-zero payoffs only) ──────────────
ax = axes[1, 0]
nonzero_ko = p_KO[p_KO > 0]
ax.set_title(f"KO put payoff — non-zero payoffs only  ({len(nonzero_ko):,} paths = {len(nonzero_ko)/N_SIM*100:.2f}%)")
if len(nonzero_ko) > 0:
    ax.hist(nonzero_ko, bins=60, color=GREEN, alpha=0.8, density=True)
    ax.axvline(nonzero_ko.mean(), color=YELLOW, lw=2, ls="--",
               label=f"Mean payoff = {nonzero_ko.mean():.3f}")
    ax.set_xlabel("Payoff per unit (when option survives)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
else:
    ax.text(0.5, 0.5, "No non-zero payoffs", transform=ax.transAxes, ha="center")
pct_zero = (p_KO == 0).mean() * 100
ax.text(0.02, 0.95, f"{pct_zero:.1f}% of paths pay 0\n(knocked out or OTM)",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round", fc="#16213e", ec=RED, alpha=0.8))
ax.grid(True, lw=0.4)

# ── Bottom-right: KO fair value vs barrier (line chart) ──────────────────────
ax = axes[1, 1]
ax.set_title("KO fair value vs assumed barrier  (vs market bid=0.15, ask=0.175)")
barriers = np.arange(25, 56, 1)
ko_fairs = []
for b in barriers:
    breach_b = np.any(prices[:, :T3_STEPS - 1] < b, axis=1)
    ko_fairs.append(float(np.mean(np.where(breach_b, 0.0, np.maximum(45 - S_3wk, 0)))))
ax.plot(barriers, ko_fairs, color=CYAN, lw=2, marker="o", ms=4, label="KO fair value")
ax.axhline(0.15,  color=GREEN, lw=1.5, ls="--", alpha=0.9, label="Market bid = 0.15")
ax.axhline(0.175, color=ORANGE, lw=1.5, ls="--", alpha=0.9, label="Market ask = 0.175")
ax.axvline(45, color=RED, lw=1.5, ls=":", alpha=0.8, label="Assumed barrier=45")

# shade: SELL zone (fair < bid), BUY zone (fair > ask)
ax.fill_between(barriers, ko_fairs, 0.15,
                where=[f < 0.15 for f in ko_fairs],
                color=RED, alpha=0.15, label="Overpriced (SELL zone)")
ax.fill_between(barriers, ko_fairs, 0.175,
                where=[f > 0.175 for f in ko_fairs],
                color=GREEN, alpha=0.15, label="Underpriced (BUY zone)")
ax.set_xlabel("Knock-out barrier")
ax.set_ylabel("KO fair value")
ax.legend(fontsize=7)
ax.grid(True, lw=0.4)

plt.tight_layout()
fig2.savefig("fig2_ko_analysis.png", dpi=150, bbox_inches="tight")
print("  Saved fig2_ko_analysis.png")

# ============================================================
# FIGURE 3 — Individual trade PnL distributions
# ============================================================
print("Plotting Figure 3: Individual trade PnL distributions...")
fig3, axes = plt.subplots(2, 3, figsize=(16, 9))
fig3.suptitle("Figure 3  |  Per-Trade PnL Distribution  (at expiry, full volume, x contract size 3000)")
axes_flat = axes.flatten()

ko_fair_mc = float(np.mean(p_KO))
trades = [
    ("BUY 50 x AC_50_P2 @ 9.75",   pnl_P2,  CYAN,   9.8707,     9.75,  "BUY"),
    ("BUY 50 x AC_50_C2 @ 9.75",   pnl_C2,  PURPLE, 9.8707,     9.75,  "BUY"),
    ("SELL 50 x chooser @ 22.20",   pnl_CO,  GREEN,  21.8977,    22.20, "SELL"),
    ("SELL 50 x binary @ 5.00",     pnl_BP,  YELLOW, 4.7679,     5.00,  "SELL"),
    ("BUY 500 x KO @ 0.175",        pnl_KO,  CYAN,   ko_fair_mc, 0.175, "BUY"),
]

for ax, (title, pnl, col, fair, entry, direction) in zip(axes_flat, trades):
    stats = risk_stats(pnl)
    clip_lo = np.percentile(pnl, 0.5)
    clip_hi = np.percentile(pnl, 99.5)
    clipped = np.clip(pnl, clip_lo, clip_hi)
    ax.hist(clipped, bins=120, color=col, alpha=0.7, density=True)
    ax.axvline(stats["E[PnL]"],  color=YELLOW, lw=2.0, ls="-",  label=f"E[PnL]  = {stats['E[PnL]']:,.0f}")
    ax.axvline(stats["VaR 5%"],  color=RED,    lw=1.5, ls="--", label=f"VaR 5%  = {stats['VaR 5%']:,.0f}")
    ax.axvline(stats["CVaR 5%"], color=ORANGE, lw=1.5, ls=":",  label=f"CVaR 5% = {stats['CVaR 5%']:,.0f}")
    ax.axvline(0, color="white", lw=1.0, ls="-", alpha=0.4)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("PnL")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
    ax.text(0.02, 0.96,
            f"P(loss) = {stats['P(loss)']*100:.1f}%\nStd     = {stats['Std']:,.0f}",
            transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(boxstyle="round", fc="#16213e", ec=col, alpha=0.8))
    ax.grid(True, lw=0.4)

axes_flat[5].set_visible(False)   # 6th panel unused

plt.tight_layout()
fig3.savefig("fig3_trade_pnl.png", dpi=150, bbox_inches="tight")
print("  Saved fig3_trade_pnl.png")

# ============================================================
# FIGURE 4 — Portfolio risk & hedging comparison
# ============================================================
print("Plotting Figure 4: Portfolio risk and hedging...")
fig4, axes = plt.subplots(2, 2, figsize=(14, 10))
fig4.suptitle("Figure 4  |  Portfolio Risk & KO Hedging Strategies")

portfolios = [
    ("No KO  (4 trades: P2+C2+CO+BP)",
     pnl_base,   CYAN),
    ("Full (all 5 trades incl. BUY 500 KO @ 0.175)",
     pnl_full,   ORANGE),
    ("Hedge A: full + SELL 500 x AC_45_P @ 9.05\n(offsets KO vanilla-put exposure)",
     pnl_hedgeA, GREEN),
    ("Hedge B: full + BUY 500 x AC_35_P @ 4.35\n(barrier insurance: pays when KO knocked out)",
     pnl_hedgeB, PURPLE),
]

for ax, (title, pnl, col) in zip(axes.flatten(), portfolios):
    stats = risk_stats(pnl)
    lo = np.percentile(pnl, 0.5)
    hi = np.percentile(pnl, 99.5)
    ax.hist(np.clip(pnl, lo, hi), bins=150, color=col, alpha=0.7, density=True)
    ax.axvline(stats["E[PnL]"],  color=YELLOW, lw=2.0, ls="-",
               label=f"E[PnL]  = {stats['E[PnL]']:>10,.0f}")
    ax.axvline(stats["VaR 5%"],  color=RED,    lw=1.5, ls="--",
               label=f"VaR 5%  = {stats['VaR 5%']:>10,.0f}")
    ax.axvline(stats["CVaR 5%"], color=ORANGE, lw=1.5, ls=":",
               label=f"CVaR 5% = {stats['CVaR 5%']:>10,.0f}")
    ax.axvline(0, color="white", lw=1.0, ls="-", alpha=0.4)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Portfolio PnL")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7, loc="upper right")
    ax.text(0.02, 0.96,
            f"P(loss) = {stats['P(loss)']*100:.1f}%\nStd     = {stats['Std']:>10,.0f}",
            transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(boxstyle="round", fc="#16213e", ec=col, alpha=0.8))
    ax.grid(True, lw=0.4)

plt.tight_layout()
fig4.savefig("fig4_portfolio_risk.png", dpi=150, bbox_inches="tight")
print("  Saved fig4_portfolio_risk.png")

# ============================================================
# FIGURE 5 — Fair value vs time (theta decay)
# ============================================================
print("Plotting Figure 5: Option fair prices vs time...")
from scipy.stats import norm as _norm

def _bs_call(S, K, T, sigma):
    if T <= 0: return max(S - K, 0.0)
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * _norm.cdf(d1) - K * _norm.cdf(d2)

def _bs_put(S, K, T, sigma):
    if T <= 0: return max(K - S, 0.0)
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * _norm.cdf(-d2) - S * _norm.cdf(-d1)

def _bs_binary_put(S, K, T, sigma, payoff=10.0):
    if T <= 0: return payoff if S < K else 0.0
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return payoff * _norm.cdf(-d2)

TRADING_DAYS = 15   # total 3-week horizon in trading days
times = np.linspace(0, TRADING_DAYS, 300)   # days elapsed from now
sigma = SIGMA

p2_fair  = [_bs_put(S0, 50, max(0, (10 - t)) / TRADING_DAYS_PER_YEAR, sigma) for t in times]
c2_fair  = [_bs_call(S0, 50, max(0, (10 - t)) / TRADING_DAYS_PER_YEAR, sigma) for t in times]
co_fair  = [_bs_call(S0, 50, max(0, (15 - t)) / TRADING_DAYS_PER_YEAR, sigma)
            + _bs_put(S0, 50, max(0, (10 - t)) / TRADING_DAYS_PER_YEAR, sigma) for t in times]
bp_fair  = [_bs_binary_put(S0, 40, max(0, (15 - t)) / TRADING_DAYS_PER_YEAR, sigma) for t in times]
ko_fair_t = [float(np.mean(p_KO)) if t == 0 else float(np.mean(p_KO) * max(0, 1 - t / TRADING_DAYS) ** 0.3)
             for t in times]  # rough KO decay proxy

fig5, ax5 = plt.subplots(figsize=(14, 6))
fig5.suptitle("Figure 5  |  Option Fair Prices vs Time  +  Trade Entries")
ax5.set_title("Model fair values over time (holding spot fixed at S0=50)", fontsize=10)

ax5.plot(times, p2_fair,  color=CYAN,   lw=2, label="AC_50_P2 fair")
ax5.plot(times, c2_fair,  color=PURPLE, lw=2, label="AC_50_C2 fair")
ax5.plot(times, co_fair,  color=GREEN,  lw=2, label="AC_50_CO fair")
ax5.plot(times, bp_fair,  color=YELLOW, lw=2, label="AC_40_BP fair")
ax5.plot(times, ko_fair_t, color=RED,   lw=2, label="AC_45_KO fair (MC)")

ax5.axvline(10, color="white", lw=1.2, ls="--", alpha=0.6, label="Chooser decision (day 10)")
ax5.axvline(15, color=GREY,    lw=1.2, ls="--", alpha=0.6, label="3-week expiry (day 15)")

market_entries = [
    (0, p2_fair[0],  9.75,  "BUY  AC_50_P2 @ 9.75",  GREEN,  "^"),
    (0, c2_fair[0],  9.75,  "BUY  AC_50_C2 @ 9.75",  GREEN,  "^"),
    (0, co_fair[0],  22.20, "SELL AC_50_CO @ 22.20",  RED,    "v"),
    (0, bp_fair[0],  5.00,  "SELL AC_40_BP @ 5.00",   RED,    "v"),
    (0, ko_fair_t[0],0.175, "BUY  AC_45_KO @ 0.175",  GREEN,  "^"),
]
for t_entry, fair_val, mkt_price, label, col, marker in market_entries:
    ax5.scatter([t_entry], [mkt_price], color=col, marker=marker, s=100, zorder=5)
    ax5.text(t_entry + 0.1, mkt_price + 0.3, label, color=col, fontsize=7)

ax5.set_xlabel("Time from now (trading days)")
ax5.set_ylabel("Option price")
ax5.legend(fontsize=8, loc="upper right")
ax5.grid(True, lw=0.4)

plt.tight_layout()
fig5.savefig("fig5_time_option_prices_trades.png", dpi=150, bbox_inches="tight")
print("  Saved fig5_time_option_prices_trades.png")

# ============================================================
# FIGURE 6 — Option prices + trade signal markers
# ============================================================
print("Plotting Figure 6: Option prices with BUY/SELL signals...")

products_fig6 = [
    ("AC_50_P2",  9.70,  9.75,  9.8707,     "BUY",  0.1207),
    ("AC_50_C2",  9.70,  9.75,  9.8707,     "BUY",  0.1207),
    ("AC_50_CO",  22.20, 22.30, 21.8977,    "SELL", 0.3023),
    ("AC_40_BP",  5.00,  5.10,  4.7679,     "SELL", 0.2321),
    ("AC_45_KO",  0.15,  0.175, ko_fair_mc, "BUY",  ko_fair_mc - 0.175),
]

fig6, ax6 = plt.subplots(figsize=(14, 6))
fig6.suptitle("Figure 6  |  Option Prices + Trade Side (BUY/SELL)")
ax6.set_title("Each product: bid/ask/fair and recommended trade side", fontsize=10)

x_pos = np.arange(len(products_fig6))
for i, (name, bid, ask, fair, action, edge) in enumerate(products_fig6):
    ax6.scatter([i], [bid],  color=CYAN,   s=60, zorder=4, label="Bid" if i == 0 else "")
    ax6.scatter([i], [ask],  color=ORANGE, s=60, zorder=4, label="Ask" if i == 0 else "")
    ax6.scatter([i], [fair], color=YELLOW, s=80, marker="D", zorder=5, label="Fair value" if i == 0 else "")

    col    = GREEN if action == "BUY" else RED
    marker = "^"   if action == "BUY" else "v"
    trade_price = ask if action == "BUY" else bid
    ax6.scatter([i], [trade_price], color=col, marker=marker, s=150, zorder=6)
    ax6.text(i, trade_price + abs(fair) * 0.04,
             f"edge={edge:+.3f}\n{action}", color=col, fontsize=8, ha="center", va="bottom")

ax6.set_xticks(x_pos)
ax6.set_xticklabels([p[0] for p in products_fig6], fontsize=9)
ax6.set_ylabel("Option price")
ax6.legend(fontsize=9, loc="upper right")
ax6.grid(True, lw=0.4)

plt.tight_layout()
fig6.savefig("fig6_option_prices_with_trades.png", dpi=150, bbox_inches="tight")
print("  Saved fig6_option_prices_with_trades.png")

# ============================================================
# Print risk table to console
# ============================================================
print()
print("=" * 80)
print("  PORTFOLIO RISK SUMMARY")
print(f"  {'Strategy':<45} {'E[PnL]':>10} {'Std':>10} {'VaR 5%':>10} {'CVaR 5%':>10} {'P(loss)':>8}")
print("-" * 80)
for title, pnl, _ in portfolios:
    s = risk_stats(pnl)
    short = title.split("\n")[0][:44]
    print(f"  {short:<45} {s['E[PnL]']:>10,.0f} {s['Std']:>10,.0f} "
          f"{s['VaR 5%']:>10,.0f} {s['CVaR 5%']:>10,.0f} {s['P(loss)']*100:>7.1f}%")
print()

# KO position size sensitivity
print("  KO SIZE SENSITIVITY  (base portfolio PnL is fixed; KO fraction varies)")
print(f"  {'KO fraction':<15} {'E[PnL]':>10} {'VaR 5%':>10} {'CVaR 5%':>10} {'P(loss)':>8}")
print("-" * 80)
for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
    pnl_scaled = pnl_base + frac * pnl_KO
    s = risk_stats(pnl_scaled)
    print(f"  {frac*100:>5.0f}% of 500       {s['E[PnL]']:>10,.0f} "
          f"{s['VaR 5%']:>10,.0f} {s['CVaR 5%']:>10,.0f} {s['P(loss)']*100:>7.1f}%")
print("=" * 80)
print("\nAll figures saved to Rounds/4_round/manual/")
