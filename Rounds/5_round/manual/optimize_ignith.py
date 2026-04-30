# -*- coding: utf-8 -*-
"""
Round 5 Manual: IGNITH EXCHANGE OPTIMISER  (CVXPY edition)
===========================================================

Utility function (from competition theory slides):

    U(pi) = r^T pi * W  -  lambda * sum(pi_i^2)

    pi_i  in R        signed allocation percentage (+ve = BUY, -ve = SELL)
    r_i   in R        signed expected return        (+ve = price up, -ve = price down)
    W     = 1,000,000 trading capital (XIRECs)
    lambda            quadratic fee coefficient

Constraint: sum(|pi_i|) <= 100   (total capital usage)

Fee formula from competition rules:
    fee_i = (|pi_i| / 100)^2 * W  =  (W/10000) * pi_i^2  =  100 * pi_i^2

=>  lambda_correct = 100   (implied by fee formula + W = 1,000,000)

The slides show lambda = 120 — we compare both here.

Analytical solution (unconstrained):
    d/d(pi_i) [10000 r_i pi_i - lambda pi_i^2] = 0
    =>  pi_i* = (10000 / 2*lambda) * r_i = (5000 / lambda) * r_i

    lambda=100  =>  pi_i* = 50 * r_i      (our base case)
    lambda=120  =>  pi_i* = 41.67 * r_i   (their formulation)

Confidence-adjusted allocation
-------------------------------
Under directional uncertainty with probability P of being right:

    E[net_i(pi)]  =  (2P-1) * (pi/100)*W*r  -  (pi/100)^2 * W
    d/dpi = 0  =>  pi_i* = 50 * r_i * (2*P_i - 1)

This is the Kelly-criterion-equivalent for this fee structure.
P=50% => pi*=0 (don't trade).  P<50% => flip direction.

Usage
-----
    python optimize_ignith.py             # confidence-adjusted (recommended)
    python optimize_ignith.py --full      # full-confidence allocations
    python optimize_ignith.py --quiet     # submission values only
    python optimize_ignith.py --input     # interactive: enter your own confidence values
"""

import sys
import io

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

QUIET       = "--quiet" in sys.argv or "-q" in sys.argv
FULL        = "--full"  in sys.argv or "-f" in sys.argv
INTERACTIVE = "--input" in sys.argv or "-i" in sys.argv

# --preset <name>  e.g.  python optimize_ignith.py --preset hedged
_preset_arg = None
if "--preset" in sys.argv:
    _idx = sys.argv.index("--preset")
    if _idx + 1 < len(sys.argv):
        _preset_arg = sys.argv[_idx + 1].lower()

import numpy as np
import cvxpy as cp

# ============================================================================
# 1. PROBLEM SETUP
# ============================================================================

BUDGET = 1_000_000   # XIRECs

# PRODUCTS tuple: (name, short, r_signed, confidence_P, signal_strength, catalyst)
#
# r_signed   : signed expected return (+ = BUY, - = SELL), magnitude from news analysis
# confidence : P(direction correct), 0.0–1.0
#              Edit these values to reflect your own conviction.
#
#   P = 0.95  Hard regulatory/government action       → nearly certain
#   P = 0.85  Clear fundamental supply/demand shock   → high confidence
#   P = 0.80  Institutional/mechanical catalyst       → high confidence
#   P = 0.75  Consumer sentiment / M&A               → moderate-high
#   P = 0.70  Forward forecast (not confirmed data)  → moderate
#   P = 0.65  Influencer-driven demand               → moderate
#   P = 0.55  Ambiguous / PR-only signal             → weak

PRODUCTS = [
    # name,               short,  r_signed,  P,     strength,    catalyst
    ("Lava Cake",         "LC",   -0.40,     0.95,  "VERY HIGH", "Sales halted; actual lava contamination + lawsuits"),
    ("Obsidian Cutlery",  "OC",   +0.30,     0.85,  "HIGH",      "Manufacturing fully halted; assembly line destroyed"),
    ("Pyroflex Cells",    "PC",   -0.25,     0.85,  "HIGH",      "40% VAT cut abruptly ended; consumer fees doubled"),
    ("Magma Ink",         "MI",   +0.25,     0.75,  "HIGH",      "6-hour queue hot drop; M&A merger confirmed"),
    ("Sulfur Reactor",    "SR",   +0.20,     0.80,  "HIGH",      "Index inclusion confirmed; passive funds must buy"),
    ("Thermalite Core",   "TC",   +0.20,     0.70,  "MOD-HIGH",  "Quarterly forecast: users 1.42M -> 3.99M"),
    ("Scoria Paste",      "SP",   +0.15,     0.65,  "MODERATE",  "Influencer stockpiling call; infrastructure staple"),
    ("Volcanic Incense",  "VI",   +0.15,     0.65,  "MODERATE",  "Ongoing rally + Profit Nostradamus public call"),
    ("Ashes of Phoenix",  "AP",   -0.10,     0.55,  "LOW",       "PR controversy; CEO damage control; boycott risk"),
]

N          = len(PRODUCTS)
NAMES      = [p[0] for p in PRODUCTS]
SHORTS     = [p[1] for p in PRODUCTS]
R          = np.array([p[2] for p in PRODUCTS])   # signed expected returns
CONFIDENCE = np.array([p[3] for p in PRODUCTS])   # P(direction correct)

# Confidence-adjusted effective returns: r_eff_i = r_i * (2*P_i - 1)
R_EFF = R * (2 * CONFIDENCE - 1)

# ============================================================================
# PRESETS
# Products order: Lava Cake, Obsidian Cutlery, Pyroflex Cells, Magma Ink,
#                 Sulfur Reactor, Thermalite Core, Scoria Paste,
#                 Volcanic Incense, Ashes of Phoenix
# ============================================================================

PRESETS = {
    "full": {
        "desc":       "Confidence-adjusted defaults — our base analysis (P = 55–95%)",
        "returns":    R.copy(),
        "confidence": CONFIDENCE.copy(),
    },
    "hedged": {
        "desc":       "History-calibrated — returns adjusted from last year's results (P = 100%)",
        # Key changes vs 'full':
        #   Lava Cake ↑ (-40% → -55%)  : safety crises over-deliver historically
        #   Obsidian  ↓ (+30% → +23%)  : supply shock still big but trimmed
        #   Pyroflex  ↓ (-25% → -18%)  : tax changes under-deliver historically
        #   Magma Ink    skipped (0%)   : M&A signal went wrong direction last year
        #   Sulfur    ↓ (+20% → +9%)   : trimmed; index timing uncertain
        #   Thermalite ↓(+20% → +17%)  : quarterly reports over-estimated last year
        #   Scoria    ↓ (+15% → +5%)   : hype articles dangerous
        #   Volcanic Incense → SELL -20%: pump-and-dump risk (Nostradamus unreliable)
        #   Ashes     ↑ (-10% → -17%)  : slightly more conviction on PR damage
        "returns":    np.array([-0.55, +0.23, -0.18,  0.00, +0.09, +0.17, +0.05, -0.20, -0.17]),
        "confidence": np.ones(9),
    },
}

PRESET_NAMES = list(PRESETS.keys())


# ============================================================================
# 2. CVXPY SOLVER
# ============================================================================

def solve(r: np.ndarray, lam: float, budget: float = BUDGET) -> np.ndarray:
    """
    Solve:  max  (W/100) * r^T pi  -  lam * ||pi||^2
            s.t. ||pi||_1 <= 100

    Returns pi* as an array of signed percentages.
    Note: (W/100) * r^T pi  is the gross profit in XIRECs
          lam * ||pi||^2     is the quadratic penalty (fee if lam=W/10000=100)
    """
    pi   = cp.Variable(N)
    obj  = cp.Maximize((budget / 100) * r @ pi - lam * cp.sum_squares(pi))
    cons = [cp.norm1(pi) <= 100]
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.CLARABEL, verbose=False)
    if pi.value is None:
        prob.solve(solver=cp.SCS, verbose=False)
    return np.array(pi.value, dtype=float)


def actual_pnl(pi: np.ndarray, r: np.ndarray, budget: float = BUDGET) -> np.ndarray:
    """
    Compute ACTUAL net PnL vector using the real fee formula (lambda=100),
    regardless of which lambda was used to optimise.
    """
    gross = (pi / 100) * budget * r          # (pi/100)*W*r per product
    fee   = (pi / 100) ** 2 * budget         # (pi/100)^2 * W per product
    return gross - fee                        # net per product


# ============================================================================
# 3. SOLVE FOR BOTH LAMBDA VALUES
# ============================================================================

LAM_OURS   = 100   # correct: implied by fee = (p/100)^2 * W = 100 * p^2
LAM_THEIRS = 120   # from the competition slides

# Full-confidence solutions (as before)
pi_ours   = solve(R,     LAM_OURS)
pi_theirs = solve(R,     LAM_THEIRS)

# Confidence-adjusted solution (EV-optimal under directional uncertainty)
# Uses R_EFF = R * (2P - 1) as the effective return vector.
pi_conf   = solve(R_EFF, LAM_OURS)

net_ours   = actual_pnl(pi_ours,   R)
net_theirs = actual_pnl(pi_theirs, R)

# For confidence-adjusted:
#   net_conf_if_right  = actual PnL if all directions are correct
#   net_conf_expected  = expected PnL accounting for P (E[net_i] = (2P-1)*gross - fee)
net_conf_if_right = actual_pnl(pi_conf, R)
net_conf_expected = np.array([
    (2*CONFIDENCE[i] - 1) * (pi_conf[i]/100) * BUDGET * R[i]
    - (pi_conf[i]/100)**2 * BUDGET
    for i in range(N)
])

# Choose which solution to use as the submission recommendation
pi_submit  = pi_conf if not FULL else pi_ours
net_submit = net_conf_if_right if not FULL else net_ours
label      = "CONFIDENCE-ADJUSTED (recommended)" if not FULL else "FULL CONFIDENCE"


# ============================================================================
# 4. ANALYTICAL CROSS-CHECK  (pi* = 5000/lambda * r, scaled if sum>100)
# ============================================================================

def analytical(r: np.ndarray, lam: float) -> np.ndarray:
    raw   = (5000 / lam) * r                    # signed, unconstrained
    total = np.abs(raw).sum()
    if total > 100:
        raw *= 100 / total
    return raw

pi_analytical = analytical(R, LAM_OURS)


# ============================================================================
# 5. PRINTING HELPERS
# ============================================================================

W = 118

def sep(c="-", width=W): print(c * width)
def hdr(t):              print(); print("=" * W); print(f"  {t}"); print("=" * W)


def print_comparison_table():
    hdr("CVXPY OPTIMISATION — lambda=100 (ours) vs lambda=120 (their slides)")
    print(f"\n  Utility function:  U(pi) = (W/100) * r^T pi  -  lambda * sum(pi_i^2)")
    print(f"  Constraint:        sum(|pi_i|) <= 100")
    print(f"  Budget W:          {BUDGET:,} XIRECs")
    print(f"  Actual fee:        (|pi_i|/100)^2 * W  =  100 * pi_i^2   [lambda_correct = 100]")
    print()

    col = "{:<22} {:>6} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}"
    sep()
    print(col.format(
        "Product", "r",
        "pi*(l=100)", "Net(l=100)",
        "pi*(l=120)", "Net(l=120)",
        "Analytic",  "Diff"
    ))
    sep()

    for i in range(N):
        diff = net_ours[i] - net_theirs[i]
        sign = "+" if diff >= 0 else ""
        print(col.format(
            NAMES[i],
            f"{R[i]:+.0%}",
            f"{pi_ours[i]:+.1f}%",
            f"{net_ours[i]:,.0f}",
            f"{pi_theirs[i]:+.1f}%",
            f"{net_theirs[i]:,.0f}",
            f"{pi_analytical[i]:+.1f}%",
            f"{sign}{diff:,.0f}",
        ))

    sep()
    total_ours   = net_ours.sum()
    total_theirs = net_theirs.sum()
    total_alloc_ours   = np.abs(pi_ours).sum()
    total_alloc_theirs = np.abs(pi_theirs).sum()
    diff_total = total_ours - total_theirs
    sign_total = "+" if diff_total >= 0 else ""
    print(col.format(
        "TOTAL", "",
        f"{total_alloc_ours:.1f}%",
        f"{total_ours:,.0f}",
        f"{total_alloc_theirs:.1f}%",
        f"{total_theirs:,.0f}",
        "100.0%",
        f"{sign_total}{diff_total:,.0f}",
    ))
    sep()

    print(f"\n  lambda=100 (ours)  :  budget used = {total_alloc_ours:.1f}%   net PnL = {total_ours:,.0f} XIRECs")
    print(f"  lambda=120 (theirs):  budget used = {total_alloc_theirs:.1f}%   net PnL = {total_theirs:,.0f} XIRECs")
    print(f"\n  Our lambda=100 outperforms by  {diff_total:,.0f} XIRECs  "
          f"({diff_total/total_theirs*100:.1f}% more PnL)")
    print(f"\n  Reason: lambda=120 under-allocates by {100-total_alloc_theirs:.1f}% of budget,")
    print(f"  leaving {(100-total_alloc_theirs)/100*BUDGET:,.0f} XIRECs of capital unused and earning nothing.")


def print_formula_derivation():
    hdr("ANALYTICAL DERIVATION  (why pi* = 50*r for lambda=100)")
    print()
    print("  Per-product net PnL:")
    print("    net_i(pi)  =  (W/100) * r_i * pi_i  -  lambda * pi_i^2")
    print()
    print("  First-order condition:")
    print("    d(net_i)/d(pi_i)  =  (W/100) * r_i  -  2 * lambda * pi_i  =  0")
    print()
    print("    =>  pi_i*  =  W * r_i / (200 * lambda)")
    print()
    print(f"    lambda=100 :  pi_i*  =  {BUDGET} * r_i / (200*100)  =  50 * r_i")
    print(f"    lambda=120 :  pi_i*  =  {BUDGET} * r_i / (200*120)  =  41.67 * r_i")
    print()
    print("  Net PnL at optimum:")
    print("    net_i(pi_i*)  =  W * r_i^2 / (200*lambda)  -  lambda * (W*r_i/(200*lambda))^2")
    print("                  =  W * r_i^2 * [1/(200*lambda)  -  lambda/(200*lambda)^2]")
    print("                  =  W * r_i^2 / (400 * lambda)")
    print()
    # actual PnL = W^2 * sum(r^2) / (40000 * lambda)
    lam_ours_pnl   = BUDGET**2 * np.sum(R**2) / (40000 * LAM_OURS)
    lam_theirs_pnl = BUDGET**2 * np.sum(R**2) / (40000 * LAM_THEIRS)
    print(f"    lambda=100 :  net  =  W^2 * sum(r^2) / 40000  =  {lam_ours_pnl:,.0f} XIRECs")
    print(f"    lambda=120 :  actual net (real fees) = {net_theirs.sum():,.0f} XIRECs  "
          f"[under-allocates => sub-optimal]")
    print(f"\n    sum(r^2)   =  {np.sum(R**2):.4f}   (drives all PnL)")


def print_sensitivity():
    hdr("SENSITIVITY ANALYSIS  (lambda=100, all returns scaled by factor X)")
    print(f"  {'Scale':>8}  {'Budget%':>8}  {'Net PnL':>14}  {'ROI':>8}")
    sep("-", 45)
    for sf in [0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00]:
        r_scaled = R * sf
        pi_s     = solve(r_scaled, LAM_OURS)
        net_s    = actual_pnl(pi_s, r_scaled)
        marker   = "  <-- base case" if sf == 1.00 else ""
        print(f"  {sf:>8.2f}  {np.abs(pi_s).sum():>8.1f}%  "
              f"{net_s.sum():>14,.0f}  {net_s.sum()/BUDGET*100:>7.2f}%{marker}")


def print_worst_case(r: np.ndarray = None, conf: np.ndarray = None):
    """Worst-case loss table for the chosen preset."""
    _r    = r    if r    is not None else R
    _conf = conf if conf is not None else CONFIDENCE
    _r_eff = _r * (2 * _conf - 1)
    _pi   = solve(_r_eff, LAM_OURS)   # confidence-adjusted allocations

    hdr("WORST-CASE  (all directions exactly wrong)")
    col = "{:<22} {:>8} {:>11} {:>12} {:>12}"
    sep("-", 70)
    print(col.format("Product", "Alloc %", "Fee", "Gross loss", "Net (loss)"))
    sep("-", 70)
    total_loss  = 0
    total_fee   = 0
    total_alloc = 0
    for i in range(N):
        p   = _pi[i]
        if abs(p) < 0.01:
            continue
        gp  = -(abs(p) / 100) * BUDGET * abs(_r[i])   # wrong direction: full move against us
        fee = (p / 100) ** 2 * BUDGET
        net = gp - fee
        total_loss  += net
        total_fee   += fee
        total_alloc += abs(p)
        print(col.format(
            NAMES[i],
            f"{p:+.1f}%",
            f"{fee:,.0f}",
            f"{gp:,.0f}",
            f"{net:,.0f}",
        ))
    sep("-", 70)
    print(col.format("TOTAL", f"{total_alloc:.1f}%", f"{total_fee:,.0f}", "", f"{total_loss:,.0f}"))
    print(f"\n  Maximum possible loss if every signal is wrong: {total_loss:,.0f} XIRECs")


def print_submission(r: np.ndarray = None, conf: np.ndarray = None,
                     preset_name: str = None):
    """Print final submission block.

    If r / conf are provided they override the module-level globals so the
    output reflects the chosen preset rather than the hard-coded defaults.
    """
    # ── resolve values ────────────────────────────────────────────────────
    _r    = r    if r    is not None else R
    _conf = conf if conf is not None else CONFIDENCE
    _full = FULL or (np.allclose(_conf, 1.0) and np.allclose(_r, R))

    _r_eff = _r * (2 * _conf - 1)
    _pi_full = solve(_r, LAM_OURS)          # full-confidence allocations
    _pi_sub  = solve(_r_eff, LAM_OURS)      # confidence-adjusted allocations
    _net_sub = actual_pnl(_pi_sub, _r)
    _net_exp = np.array([
        (2*_conf[i] - 1) * (_pi_sub[i]/100) * BUDGET * _r[i]
        - (_pi_sub[i]/100)**2 * BUDGET
        for i in range(N)
    ])
    _pi_analytical = analytical(_r, LAM_OURS)
    _max_diff = np.max(np.abs(_pi_full - _pi_analytical))

    # display label
    if preset_name:
        lbl = preset_name.upper()
    elif _full:
        lbl = "FULL CONFIDENCE"
    else:
        lbl = "CONFIDENCE-ADJUSTED"

    hdr(f"FINAL SUBMISSION  —  {lbl}")
    print()
    if not _full:
        print(f"  Using confidence-adjusted allocations: pi* = 50 * r * (2P - 1)")
        print(f"  Override with --full to use full-confidence allocations.\n")

    print(f"  {'Product':<22}  {'Direction':<6}  {'Alloc %':>7}  {'Bar'}")
    sep("-", 70)
    for i in range(N):
        p    = _pi_sub[i]
        pf   = _pi_full[i]
        d    = "BUY" if p > 0 else "SELL"
        bar  = "#" * max(0, int(abs(p) / 1.5))
        diff = abs(p) - abs(pf)
        note = f"  ({diff:+.1f}% vs full)" if abs(diff) > 0.05 else ""
        print(f"  {NAMES[i]:<22}  {d:<6}  {abs(p):>6.1f}%  {bar}{note}")
    sep("-", 70)
    total_alloc = np.abs(_pi_sub).sum()
    total_net   = _net_sub.sum()
    print(f"  {'Total budget used':<30}: {total_alloc:.1f}%")
    print(f"  {'PnL if all directions correct':<30}: {total_net:,.0f} XIRECs")
    if not _full:
        print(f"  {'E[PnL] under uncertainty':<30}: {_net_exp.sum():,.0f} XIRECs")
    print(f"  {'ROI (if correct)':<30}: {total_net/BUDGET*100:.2f}%")
    sep("-", 70)
    print(f"\n  CVXPY solver        : CLARABEL (convex QP)")
    print(f"  Analytical check    : max|pi_cvxpy - pi_analytic| = {_max_diff:.4f}%")


# ============================================================================
# 6. RESULTS TABLE
#    Fill ACTUAL_RETURNS below once the round resolves.
#    Leave as None to show "TBD".
# ============================================================================

ACTUAL_RETURNS = [
    None,   # Lava Cake
    None,   # Obsidian Cutlery
    None,   # Pyroflex Cells
    None,   # Magma Ink
    None,   # Sulfur Reactor
    None,   # Thermalite Core
    None,   # Scoria Paste
    None,   # Volcanic Incense
    None,   # Ashes of Phoenix
]


def print_confidence_analysis(r: np.ndarray = None, conf: np.ndarray = None):
    """Show full vs confidence-adjusted comparison for the chosen preset."""
    _r    = r    if r    is not None else R
    _conf = conf if conf is not None else CONFIDENCE
    _r_eff  = _r * (2 * _conf - 1)
    _pi_full = solve(_r,     LAM_OURS)   # full-confidence
    _pi_conf = solve(_r_eff, LAM_OURS)   # confidence-adjusted
    _net_full = actual_pnl(_pi_full, _r)
    _net_conf_exp = np.array([
        (2*_conf[i] - 1) * (_pi_conf[i]/100) * BUDGET * _r[i]
        - (_pi_conf[i]/100)**2 * BUDGET
        for i in range(N)
    ])

    hdr("CONFIDENCE-ADJUSTED RISK MANAGEMENT")
    print()
    print("  Formula:  pi_i* = 50 * r_i * (2*P_i - 1)")
    print("  P = probability of being right about direction.")
    print("  P <= 50%  =>  pi* = 0 or flipped  (don't trade / bet opposite).")
    print()

    col = "{:<22} {:>6} {:>6} {:>6} {:>12} {:>12} {:>12} {:>12} {:>12}"
    sep("-", 108)
    print(col.format(
        "Product", "r", "P", "2P-1",
        "pi*(full)", "pi*(conf)",
        "E[PnL] full", "E[PnL] conf", "Saved%"
    ))
    sep("-", 108)

    total_full_if_right = 0.0
    total_conf_expected = 0.0
    total_alloc_full    = 0.0
    total_alloc_conf    = 0.0
    total_wc_full       = 0.0
    total_wc_conf       = 0.0

    for i in range(N):
        ri  = _r[i]
        P   = _conf[i]
        k   = 2*P - 1
        pf  = _pi_full[i]
        pc  = _pi_conf[i]
        ef  = _net_full[i]
        ec  = _net_conf_exp[i]
        wc_full = -(abs(pf)/100)*BUDGET*abs(ri) - (pf/100)**2*BUDGET
        wc_conf = -(abs(pc)/100)*BUDGET*abs(ri) - (pc/100)**2*BUDGET
        saved   = abs(pf) - abs(pc)

        total_full_if_right += ef
        total_conf_expected += ec
        total_alloc_full    += abs(pf)
        total_alloc_conf    += abs(pc)
        total_wc_full       += wc_full
        total_wc_conf       += wc_conf

        print(col.format(
            NAMES[i],
            f"{ri:+.0%}", f"{P:.0%}", f"{k:+.2f}",
            f"{pf:+.1f}%", f"{pc:+.1f}%",
            f"{ef:,.0f}", f"{ec:,.0f}",
            f"-{saved:.1f}%",
        ))

    sep("-", 108)
    print(col.format(
        "TOTAL", "", "", "",
        f"{total_alloc_full:.1f}%", f"{total_alloc_conf:.1f}%",
        f"{total_full_if_right:,.0f}", f"{total_conf_expected:,.0f}",
        f"-{total_alloc_full-total_alloc_conf:.1f}%",
    ))
    sep("-", 108)

    budget_saved = total_alloc_full - total_alloc_conf
    wc_saved     = total_wc_full - total_wc_conf
    pnl_cost     = total_full_if_right - total_conf_expected

    print(f"\n  ┌─ FULL CONFIDENCE  ──────────────────────────────────────────────┐")
    print(f"  │  Budget used   : {total_alloc_full:>6.1f}%                                    │")
    print(f"  │  PnL if right  : {total_full_if_right:>10,.0f} XIRECs                            │")
    print(f"  │  Worst-case    : {total_wc_full:>10,.0f} XIRECs  (all directions wrong)   │")
    print(f"  └────────────────────────────────────────────────────────────────────┘")
    print()
    print(f"  ┌─ CONFIDENCE-ADJUSTED  ──────────────────────────────────────────┐")
    print(f"  │  Budget used    : {total_alloc_conf:>6.1f}%  ({budget_saved:.1f}% freed)               │")
    print(f"  │  E[PnL]         : {total_conf_expected:>10,.0f} XIRECs  (accounts for uncertainty)  │")
    wc_pct = abs(wc_saved/total_wc_full)*100 if total_wc_full != 0 else 0
    print(f"  │  Worst-case     : {total_wc_conf:>10,.0f} XIRECs  ({wc_pct:.0f}% less downside)        │")
    print(f"  └────────────────────────────────────────────────────────────────────┘")
    print()
    print(f"  E[PnL] cost of being risk-managed : {pnl_cost:,.0f} XIRECs")
    print(f"  Worst-case improvement             : {abs(wc_saved):,.0f} XIRECs less exposure")
    print(f"\n  Run with --full to override and use full-confidence allocations.")


def print_results_table(r: np.ndarray = None, conf: np.ndarray = None):
    """Print expected vs actual table.  Accepts optional preset overrides."""
    _r    = r    if r    is not None else R
    _conf = conf if conf is not None else CONFIDENCE
    _r_eff = _r * (2 * _conf - 1)
    _pi   = solve(_r_eff, LAM_OURS)
    _net  = actual_pnl(_pi, _r)

    hdr("EXPECTED vs ACTUAL MOVEMENT")
    print()

    col = "{:<24}  {:>18}  {:>18}  {:>14}  {:>14}"
    print(col.format("Product", "Expected Movement", "Actual Movement",
                     "Our Alloc", "Est. Net PnL"))
    sep("-", 95)

    total_exp_pnl  = 0.0
    total_act_pnl  = 0.0
    any_actual     = any(a is not None for a in ACTUAL_RETURNS)

    for i in range(N):
        r_exp  = _r[i]
        r_act  = ACTUAL_RETURNS[i]
        pi_i   = _pi[i]
        d      = "BUY" if pi_i > 0 else "SELL"

        exp_pnl = _net[i]
        total_exp_pnl += exp_pnl

        if r_act is not None:
            act_net = actual_pnl(np.array([pi_i]), np.array([r_act]))[0]
            total_act_pnl += act_net
            act_str     = f"{r_act:+.2%}"
            act_pnl_str = f"{act_net:+,.0f}"
            correct = (r_act * r_exp) > 0
            flag    = "✓" if correct else "✗"
            act_str = f"{act_str} {flag}"
        else:
            act_str     = "TBD"
            act_pnl_str = "TBD"

        print(col.format(
            NAMES[i],
            f"{r_exp:+.0%}  ({d}  {abs(pi_i):.1f}%)",
            act_str,
            f"{pi_i:+.1f}%",
            act_pnl_str if r_act is not None else f"~{exp_pnl:,.0f}",
        ))

    sep("-", 95)
    exp_pnl_str = f"{total_exp_pnl:+,.0f}"
    act_pnl_str = f"{total_act_pnl:+,.0f}" if any_actual else "TBD"
    print(col.format(
        "\u03a3  (Total)",
        "",
        "",
        f"{np.abs(_pi).sum():.1f}%",
        act_pnl_str if any_actual else f"~{exp_pnl_str}",
    ))
    print()
    if any_actual:
        print(f"  Expected PnL : {total_exp_pnl:>12,.0f} XIRECs")
        print(f"  Actual PnL   : {total_act_pnl:>12,.0f} XIRECs")
        print(f"  Difference   : {total_act_pnl - total_exp_pnl:>+12,.0f} XIRECs")
    else:
        print(f"  Expected PnL : {total_exp_pnl:>12,.0f} XIRECs  (fill ACTUAL_RETURNS once round resolves)")


# ============================================================================
# 7. INTERACTIVE INPUT + SORTED ALLOCATION DISPLAY
# ============================================================================

def get_inputs_interactively():
    """
    Prompt the user to enter expected move and confidence for each product.
    Returns (returns_array, confidence_array).

    Expected move: signed % — negative = expect price to fall (SELL), positive = rise (BUY)
    Confidence:    0–100%  — probability you are right about the direction
    """
    print()
    print("=" * 78)
    print("  INTERACTIVE INPUT")
    print("  For each product, enter your expected move and confidence.")
    print("  Press Enter on any field to keep the default value.")
    print()
    print("  Expected move : signed % — use NEGATIVE for SELL, POSITIVE for BUY")
    print("                  e.g.  -40  means you expect a 40% price DROP  (SELL)")
    print("                        +25  means you expect a 25% price RISE  (BUY)")
    print("  Confidence    : 0–100% — how sure you are about the direction")
    print("                  95=near-certain  80=high  70=moderate  55=weak  50=skip")
    print("=" * 78)
    print()

    new_r    = R.copy()
    new_conf = CONFIDENCE.copy()

    for i in range(N):
        r_def = R[i]
        p_def = CONFIDENCE[i]
        d_def = "BUY" if r_def > 0 else "SELL"

        print(f"  ── {NAMES[i]}  (default: {d_def} {abs(r_def)*100:.0f}% move, {p_def*100:.0f}% confidence) ──")

        # --- Expected move ---
        while True:
            try:
                raw = input(f"     Expected move %  [{r_def:+.0%}]: ").strip()
                if raw == "":
                    break
                val = float(raw.replace("%", ""))
                new_r[i] = val / 100.0
                break
            except ValueError:
                print("     ! Enter a number, e.g. -40 or +25.")

        # --- Confidence ---
        while True:
            try:
                raw = input(f"     Confidence %     [{p_def*100:.0f}%]: ").strip()
                if raw == "":
                    break
                val = float(raw.replace("%", ""))
                if not (0 <= val <= 100):
                    print("     ! Enter a value between 0 and 100.")
                    continue
                new_conf[i] = val / 100.0
                break
            except ValueError:
                print("     ! Enter a number, e.g. 80.")

        # Show what was set
        d_new = "BUY" if new_r[i] > 0 else ("SELL" if new_r[i] < 0 else "SKIP")
        alloc = 50 * new_r[i] * (2 * new_conf[i] - 1)
        print(f"     => {d_new}  {abs(new_r[i])*100:.1f}% move  |  {new_conf[i]*100:.0f}% confidence"
              f"  |  alloc = {alloc:+.1f}%")
        print()

    return new_r, new_conf


def print_sorted_allocations(conf: np.ndarray, r: np.ndarray = R) -> None:
    """
    Given a confidence array, recompute and display allocations
    sorted from largest to smallest absolute allocation.
    """
    r_eff   = r * (2 * conf - 1)
    pi_s    = solve(r_eff, LAM_OURS)
    net_s   = actual_pnl(pi_s, r)
    e_net_s = np.array([
        (2*conf[i]-1) * (pi_s[i]/100) * BUDGET * r[i]
        - (pi_s[i]/100)**2 * BUDGET
        for i in range(N)
    ])

    # Sort by |allocation| descending
    order = np.argsort(np.abs(pi_s))[::-1]

    print()
    print("=" * 80)
    print("  SORTED ALLOCATIONS  (largest → smallest)")
    print("=" * 80)
    print()

    col = "{:<3} {:<22} {:<5} {:>6} {:>6} {:>8} {:>12} {:>12}"
    print(col.format(
        "#", "Product", "Dir", "r", "P",
        "Alloc %", "If Right", "E[PnL]"
    ))
    print("-" * 80)

    total_alloc    = 0.0
    total_if_right = 0.0
    total_expected = 0.0

    for rank, i in enumerate(order, 1):
        p   = pi_s[i]
        d   = "BUY" if p > 0 else "SELL"
        bar = "█" * max(0, int(abs(p) / 1.5))

        # skip near-zero allocations
        if abs(p) < 0.05:
            skipped_name = NAMES[i]
            print(col.format(
                f"{rank}.", skipped_name, d,
                f"{r[i]:+.0%}", f"{conf[i]:.0%}",
                "~0.0%", "—", "—"
            ) + "  (P ≈ 50%, skipped)")
            continue

        total_alloc    += abs(p)
        total_if_right += net_s[i]
        total_expected += e_net_s[i]

        print(col.format(
            f"{rank}.", NAMES[i], d,
            f"{r[i]:+.0%}", f"{conf[i]:.0%}",
            f"{p:+.1f}%",
            f"{net_s[i]:,.0f}",
            f"{e_net_s[i]:,.0f}",
        ) + f"  {bar}")

    print("-" * 80)
    print(col.format(
        "", "TOTAL", "", "", "",
        f"{total_alloc:.1f}%",
        f"{total_if_right:,.0f}",
        f"{total_expected:,.0f}",
    ))
    print()
    print(f"  Budget used      : {total_alloc:.1f}%  "
          f"({100-total_alloc:.1f}% unallocated → expires worthless)")
    print(f"  PnL if all right : {total_if_right:,.0f} XIRECs")
    print(f"  E[PnL]           : {total_expected:,.0f} XIRECs  (accounts for P per product)")
    print()
    print("  SUBMISSION VALUES:")
    print("  " + "-" * 50)
    for i in order:
        p = pi_s[i]
        if abs(p) < 0.05:
            continue
        d = "BUY" if p > 0 else "SELL"
        print(f"  {NAMES[i]:<22}  {d:<5}  {abs(p):.1f}%")
    print("  " + "-" * 50)


# ============================================================================
# 8. MAIN
# ============================================================================

def select_preset() -> dict:
    """Prompt the user to choose a preset. Returns the preset dict."""
    print()
    print("=" * 60)
    print("  SELECT PRESET")
    print("=" * 60)
    for i, (name, p) in enumerate(PRESETS.items(), 1):
        print(f"  [{i}]  {name:<10}  {p['desc']}")
    print(f"  [{len(PRESETS)+1}]  input      Enter your own values interactively")
    print()

    while True:
        raw = input(f"  Choice [1]: ").strip()
        if raw == "":
            return list(PRESETS.values())[0], list(PRESETS.keys())[0], False
        try:
            idx = int(raw)
            if idx == len(PRESETS) + 1:
                return None, "input", True    # trigger interactive
            if 1 <= idx <= len(PRESETS):
                name = PRESET_NAMES[idx - 1]
                return PRESETS[name], name, False
        except ValueError:
            if raw.lower() in PRESETS:
                return PRESETS[raw.lower()], raw.lower(), False
        print(f"  ! Enter a number between 1 and {len(PRESETS)+1}.")


def main():
    # ── Interactive mode ───────────────────────────────────────────────────
    if INTERACTIVE:
        user_r, user_conf = get_inputs_interactively()
        print_sorted_allocations(user_conf, r=user_r)
        return

    # ── Preset via --preset flag ───────────────────────────────────────────
    if _preset_arg is not None:
        if _preset_arg not in PRESETS:
            print(f"Unknown preset '{_preset_arg}'. Available: {', '.join(PRESET_NAMES)}")
            sys.exit(1)
        preset      = PRESETS[_preset_arg]
        preset_name = _preset_arg
    elif not QUIET and not FULL:
        # No flags → ask which preset
        preset, preset_name, do_interactive = select_preset()
        if do_interactive:
            user_r, user_conf = get_inputs_interactively()
            print_sorted_allocations(user_conf, r=user_r)
            return
    else:
        preset      = PRESETS["full"]
        preset_name = "full"

    use_r    = preset["returns"]
    use_conf = preset["confidence"]

    # ── Quiet mode: submission values only ────────────────────────────────
    if QUIET:
        print_sorted_allocations(use_conf, r=use_r)
        return

    # ── Full output ────────────────────────────────────────────────────────
    print()
    print("=" * W)
    print(f"  PRESET: {preset_name.upper()}  —  {preset['desc']}")
    print("=" * W)

    print_sorted_allocations(use_conf, r=use_r)

    if not FULL:
        print_confidence_analysis(r=use_r, conf=use_conf)
    print_comparison_table()
    print_sensitivity()
    print_worst_case(r=use_r, conf=use_conf)
    print_submission(r=use_r, conf=use_conf, preset_name=preset_name)
    print_results_table(r=use_r, conf=use_conf)

    print()
    print("=" * W)
    print("  Done.")
    print("=" * W)


if __name__ == "__main__":
    main()
