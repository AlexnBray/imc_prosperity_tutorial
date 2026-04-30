# Round 5 Manual: Ignith Exchange — "Extra! Extra! Read All About It!"

## Overview

Round 5 manual challenge introduces a one-day trading session on the **Ignith exchange**,
a neighbouring planet known for its volcanic scenery and "heated market dynamics."

You hold a fixed budget of **1,000,000 XIRECs** and can allocate any fraction across
9 Ignith tradable goods. You can either **BUY** (profit if price rises) or **SELL**
(profit if price falls). Unspent budget expires worthless — it does not earn a return.

The primary information source is **Ashflow Alpha**, Ignith's most trusted news outlet.
All trading decisions are based solely on the news articles published there.

---

## Tradable Goods (9 products)

| Product | Category |
|---|---|
| Obsidian Cutlery | Manufacturing / Tools |
| Pyroflex Cells | Energy / Tech |
| Thermalite Core | Smart Home / Devices |
| Lava Cake | Food / Consumer |
| Magma Ink | Stationery / Consumer |
| Scoria Paste | Infrastructure / Materials |
| Ashes of the Phoenix | Specialty / Luxury |
| Volcanic Incense | Wellness / Lifestyle |
| Sulfur Reactor | Industrial / Elemental |

---

## Rules

| Rule | Detail |
|---|---|
| Budget | 1,000,000 XIRECs |
| Max total allocation | ≤ 100% of budget |
| Unused budget | Expires worthless (does NOT add to PnL) |
| Directions | BUY (long) or SELL (short) per product |
| Allocation granularity | Percentage of budget per product |
| Fee formula | See below |

---

## Fee Formula

```
fee_i = (percentage_i / 100)² × budget
```

Where `percentage_i` is the portion of the budget (0–100) allocated to product `i`.

The fee is charged **per product**, and the total fee is the sum across all products.

**Examples:**

| Allocation | Fee |
|---|---|
| 100% into one product | (1.00)² × 1,000,000 = **1,000,000** — entire budget consumed |
| 50% into one product  | (0.50)² × 1,000,000 = **250,000** |
| 20% into one product  | (0.20)² × 1,000,000 = **40,000** |
| 10% into one product  | (0.10)² × 1,000,000 = **10,000** |
| 5% into one product   | (0.05)² × 1,000,000 = **2,500** |

The quadratic structure means **fees grow as the square of your concentration**.
Doubling your allocation quadruples the fee. This mechanically penalises
over-concentration and rewards diversification.

---

## Profit Calculation

For product `i` with `p_i` percent allocated and actual price change `r_i` (as decimal):

```
Investment_i   = (p_i / 100) × 1,000,000
Gross_profit_i = Investment_i × r_i        (positive if direction correct)
Fee_i          = (p_i / 100)² × 1,000,000
Net_profit_i   = Gross_profit_i − Fee_i
```

Total PnL = sum of all `Net_profit_i`.

---

## Information Source: Ashflow Alpha

Nine articles were published in the Ashflow Alpha edition used for this round.
Each article corresponds to exactly one of the 9 tradable goods and contains
market-moving news. The articles are:

| Article Title | Product |
|---|---|
| Crowds Line Up for Limited-Edition Lava Fountain Pen Featuring Magma Ink | Magma Ink |
| Manufacturing Halted After Obsidian Cutlery Cuts Through Its Own Assembly Line | Obsidian Cutlery |
| Ignith Tax Authority Faces Industry Pressure After Abrupt End to Pyroflex VAT Tax Cut | Pyroflex Cells |
| Quarterly Forecast Report Shows Surge in Thermalite-Powered Household Devices | Thermalite Core |
| Resurfaced Video of Ashes of the Phoenix Origin Shock Public | Ashes of the Phoenix |
| Lava D. Ray Says "Glory Days Are Ahead" for Ignith Economy, Urges Stockpiling of Scoria Paste | Scoria Paste |
| Traces of Actual Lava Found in Lava Cakes, Prompting Health Review | Lava Cake |
| Sudden Surge in Volcanic Incense as Profit Nostradamus Calls for People to Follow His Lead | Volcanic Incense |
| Index Committee Confirms Sulfur Last in Its Upcoming Review | Sulfur Reactor |

---

## Files

| File | Purpose |
|---|---|
| `README.md` | This file — round rules, products, fee structure |
| `STRATEGY.md` | Full article analysis, signal ratings, optimisation math, final allocations |
| `optimize_ignith.py` | Python script — computes optimal allocations, fees, expected PnL, sensitivity analysis |

---

## How to Run

```bash
cd Rounds/5_round/manual
python optimize_ignith.py          # default: full output + sensitivity table
python optimize_ignith.py --quiet  # submission values only
```

Requires: `numpy` only (no heavy dependencies).
