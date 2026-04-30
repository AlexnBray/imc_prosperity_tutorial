# Round 5 Manual: Strategy — Ignith Exchange

## Mathematical Framework

### Optimal Allocation Derivation

With budget B = 1,000,000 and expected return `r` (decimal) for a single product at
allocation `p` percent:

```
Net_profit(p) = (p/100) × B × r  −  (p/100)² × B
```

Differentiating with respect to `p` and setting to zero:

```
d/dp [ (p/100) × r  −  (p/100)² ]  =  0
r/100  −  2p/10000  =  0
p*  =  50 × r
```

**The unconstrained optimal allocation is `p* = 50 × r` percent.**

At optimal allocation, the net profit simplifies to:

```
Net_profit(p*)  =  B × r² / 4  =  250,000 × r²
```

This means:
- Profit scales as the **square** of the expected return
- High-conviction, large-move trades are dramatically more valuable
- Getting the direction wrong costs the fee AND the gross PnL

### Budget Constraint

If the sum of unconstrained optima exceeds 100%, scale all allocations proportionally:

```
if Σ p*_i > 100:
    p_i  →  p_i × (100 / Σ p*_i)
```

In our case the unconstrained total is exactly 100% — no scaling required.

### When to Skip a Trade

A trade with expected return `r` at its optimal allocation earns `250,000 × r²`.
It is only worth trading if you believe `r > 0` — even a small allocation to a
correct direction earns positive net profit (e.g. 5% allocation on 10% return
earns 2,500 after fees). Skip only if you have genuinely no directional view.

---

## Article Analysis

### 1. Lava Cake — SELL ★★★★★ (Strongest)

**Headline:** "Traces of Actual Lava Found in Lava Cakes, Prompting Health Review"

**Key facts:**
- Scientists discovered **actual lava** in the edible product Lava Cake
- Health authorities launched a **formal review**
- **Immediate halt in sales** ordered pending investigation
- "Potential health risks associated with volcanic material exposure"
- Lavafast (manufacturer) co-operating, but **lawyers already filing**
- Stock being returned with "severe letters"

**Analysis:**
A government-mandated sales halt is the most severe catalyst possible for
a consumer food product. Revenue goes to zero overnight. The legal exposure
adds long-term pressure beyond the immediate ban. Consumer trust is destroyed
— even after the review concludes, demand recovery will be slow. This is an
unambiguous, hard, multi-layered bearish catalyst.

| Attribute | Value |
|---|---|
| Direction | **SELL** |
| Expected return | **−40%** |
| Confidence | **VERY HIGH** |
| Optimal allocation | **20.0%** |
| Expected net PnL | **40,000 XIRECs** |

---

### 2. Obsidian Cutlery — BUY ★★★★☆

**Headline:** "Manufacturing Halted After Obsidian Cutlery Cuts Through Its Own Assembly Line"

**Key facts:**
- Large-scale manufacturing facility **suspended production**
- Serrated obsidian blades cut through the chassis assembly line's chains
- Breakdown described as "embarrassing" by company officials
- **Temporary evacuation** of the site
- Officials warn the incident "could have implications for other manufacturing facilities"
- Experts said this could affect implications for alternate manufacturing sites

**Analysis:**
A complete halt in manufacturing is a pure supply shock. Existing inventory
becomes the only available supply while demand is unchanged. The warning about
"implications for other facilities" creates additional upside risk — if the
problem propagates across the sector, the supply crunch deepens further.
Classic textbook supply disruption → price rise.

| Attribute | Value |
|---|---|
| Direction | **BUY** |
| Expected return | **+30%** |
| Confidence | **HIGH** |
| Optimal allocation | **15.0%** |
| Expected net PnL | **22,500 XIRECs** |

---

### 3. Pyroflex Cells — SELL ★★★★☆

**Headline:** "Ignith Tax Authority Faces Industry Pressure After Abrupt End to Pyroflex VAT Tax Cut"

**Key facts:**
- 40% Pyroflex Industry Tax Cut (PITC) removed **effective tomorrow**
- Cut was introduced to stimulate the "Pyroflex transition"
- Industry groups say the abrupt end "suddenly doubles current fees"
- Will "disrupt consumer upgrade cycles and slow new purchases"
- Tax Authority under cross-sector pressure but proceeding with abolishment
- Industry groups calling for reversal, but decision is final for now

**Analysis:**
The VAT cut ending tomorrow is an immediate demand-destruction event. When
effective consumer costs double overnight, purchases are deferred. "Upgrade cycles"
being disrupted means the medium-term demand pipeline shrinks. Industry pushback
confirms the market understands this is negative, but it's happening regardless.
The abruptness ("effective tomorrow") means it cannot be slowly priced in.

| Attribute | Value |
|---|---|
| Direction | **SELL** |
| Expected return | **−25%** |
| Confidence | **HIGH** |
| Optimal allocation | **12.5%** |
| Expected net PnL | **15,625 XIRECs** |

---

### 4. Magma Ink — BUY ★★★★☆

**Headline:** "Crowds Line Up for Limited-Edition Lava Fountain Pen Featuring Magma Ink"

**Key facts:**
- Limited-edition Lava Fountain Pen with built-in Magma Ink reservoir sold yesterday
- Large crowd gathered at Rock & Flow Stationery shop in Magnos Shopping Centre
- Merger between **Slip Stationery Enterprises and Splatater Inc** confirmed
  (companies behind the pen and the Magma Ink respectively)
- Several visitors waited **6+ hours** in line
- Widely promoted as a "hot drop"

**Analysis:**
Two compounding catalysts: (1) The "hot drop" format creates FOMO-driven retail
demand surge — 6-hour queues are the clearest possible demand signal. (2) The
merger between the pen and ink manufacturers validates the strategic importance
of Magma Ink to the combined entity, locking in a long-term commercial tie.
Limited edition + corporate endorsement + queuing crowds = strongly bullish.

| Attribute | Value |
|---|---|
| Direction | **BUY** |
| Expected return | **+25%** |
| Confidence | **HIGH** |
| Optimal allocation | **12.5%** |
| Expected net PnL | **15,625 XIRECs** |

---

### 5. Sulfur Reactor — BUY ★★★★☆

**Headline:** "Index Committee Confirms Sulfur Last in Its Upcoming Review"

**Key facts:**
- Elemental Index YB will **add Sulfur Reactor** in its upcoming rebalance
- Release follows a full review of constituents across the elemental products sectors
- "Sulfur Lab's flagship sulfur reactor is considered a benchmark product"
- "Funds tracking the index are expected to adjust their holdings accordingly
  once the rebalance takes effect later this cycle"

**Analysis:**
Index inclusion is one of the most mechanically reliable catalysts in financial
markets. Passive funds tracking the Elemental Index YB are *forced* to buy the
Sulfur Reactor when the rebalance hits — this is non-discretionary demand.
The phrase "later this cycle" means it is imminent, not distant. The product
is described as a "benchmark" — it already has institutional recognition. Risk:
some price appreciation may already be priced in if the news leaked early.

| Attribute | Value |
|---|---|
| Direction | **BUY** |
| Expected return | **+20%** |
| Confidence | **HIGH** |
| Optimal allocation | **10.0%** |
| Expected net PnL | **10,000 XIRECs** |

---

### 6. Thermalite Core — BUY ★★★☆☆

**Headline:** "Quarterly Forecast Report Shows Surge in Thermalite-Powered Household Devices"

**Key facts:**
- Quarterly forecast: active projected users rising from **1.42M to 3.99M** next quarter
- Nearly a **3× increase** in user base in one quarter
- Shift from short-form data specialist niche → **mainstream outsourced household use**
- "Rise in usage metrics, leading analysts to speculate about a very strong next quarter"
- Analysts explicitly bullish on the next quarter outlook

**Analysis:**
A nearly 3× user growth forecast is a strong demand signal. The shift from
niche to mainstream is a structural positive — it indicates a new and larger
addressable market being unlocked. The analyst community is on board. The
main risk is that this is a *forecast*, not a confirmed data point — forecasts
can miss. Still, the magnitude of the projected growth is hard to dismiss.

| Attribute | Value |
|---|---|
| Direction | **BUY** |
| Expected return | **+20%** |
| Confidence | **MODERATE-HIGH** |
| Optimal allocation | **10.0%** |
| Expected net PnL | **10,000 XIRECs** |

---

### 7. Scoria Paste — BUY ★★★☆☆

**Headline:** "Lava D. Ray Says 'Glory Days Are Ahead' for Ignith Economy, Urges Stockpiling of Scoria Paste"

**Key facts:**
- Lava D. Ray, "creative multimillionaire and self-proclaimed economic analyst" (BronzeTale Luke)
- Made the call during his ongoing "Ideas streaming marathon"
- Instructed followers to "stock up on Terra Picante [Scoria Paste] before it becomes unaffordable"
- Scoria Paste is "used extensively in residential repairs and infrastructure across Ignith"
- Described as "the paste that keeps Ignith together" — first-line household conditions indicator

**Analysis:**
This signal has two layers: the influencer demand pull AND the underlying
infrastructure utility. Pure influencer calls are risky (hype reverses). But
Scoria Paste has genuine structural demand — it is a maintenance staple, not a
discretionary luxury. If economic optimism improves, infrastructure spending
rises, naturally supporting Scoria Paste demand independent of the influencer.
The dual-layer reduces the fragility of this trade.

| Attribute | Value |
|---|---|
| Direction | **BUY** |
| Expected return | **+15%** |
| Confidence | **MODERATE** |
| Optimal allocation | **7.5%** |
| Expected net PnL | **5,625 XIRECs** |

---

### 8. Volcanic Incense — BUY ★★★☆☆

**Headline:** "Sudden Surge in Volcanic Incense as Profit Nostradamus Calls for People to Follow His Lead"

**Key facts:**
- Volcanic Incense "extended its rally this cycle" — **ongoing price appreciation**
- "Profit Nostradamus" made multiple public appearances publicly calling to buy
- Trading data shows a **second surge within this narrow time window**
- He calls for "genuine interest in making money" — direct retail mobilisation

**Analysis:**
Importantly, the rally is *already happening* — this is a momentum trade backed
by observed price action, not just a prediction. The influencer is amplifying
an existing trend, making the self-fulfilling prophecy more credible. "Second
surge in a narrow window" suggests the wave structure is intact. Risk: momentum
trades are inherently fragile; if the influencer's credibility wanes or the
broader market turns, reversal can be sharp.

| Attribute | Value |
|---|---|
| Direction | **BUY** |
| Expected return | **+15%** |
| Confidence | **MODERATE** |
| Optimal allocation | **7.5%** |
| Expected net PnL | **5,625 XIRECs** |

---

### 9. Ashes of the Phoenix — SELL ★★☆☆☆ (Weakest)

**Headline:** "Resurfaced Video of Ashes of the Phoenix Origin Shock Public"

**Key facts:**
- Public concern escalated after a **resurfaced video** showing supplier conditions
- Video shows birds creating paintings using flames — NOT being reduced to ashes
- CEO of Phoenix Partners Ltd immediately issued reassurance: birds are "entirely immortal"
- CEO: "Methods for Ashes of the Phoenix have been the same for millions of years"
- CEO emphasised "birds in any way" are unharmed

**Analysis:**
This is the weakest and most ambiguous signal in the dataset. The public is
"shocked" by the ORIGINS of the product — an ethical/ESG controversy. Boycott
risk from outraged consumers creates demand-side pressure downward.

However, the CEO's reassurance complicates the picture:
- "Immortal birds" → supply is NOT disrupted
- "Same methods for millions of years" → no structural change to the product

The key question is whether public outrage translates to purchasing behaviour
change. In volatile news cycles, ESG controversies typically cause a short-term
dip as reactive consumers act, then a partial recovery as the news fades.

We trade it as a weak SELL with a conservative expected return of −10%.
If uncertain, this is the one trade to reduce or skip.

| Attribute | Value |
|---|---|
| Direction | **SELL** |
| Expected return | **−10%** |
| Confidence | **LOW** |
| Optimal allocation | **5.0%** |
| Expected net PnL | **2,500 XIRECs** |

---

## Final Portfolio Summary

| Product | Direction | Exp. Return | Alloc % | Investment | Fee | Net PnL |
|---|---|---|---|---|---|---|
| Lava Cake | SELL | −40% | 20.0% | 200,000 | 40,000 | **40,000** |
| Obsidian Cutlery | BUY | +30% | 15.0% | 150,000 | 22,500 | **22,500** |
| Pyroflex Cells | SELL | −25% | 12.5% | 125,000 | 15,625 | **15,625** |
| Magma Ink | BUY | +25% | 12.5% | 125,000 | 15,625 | **15,625** |
| Sulfur Reactor | BUY | +20% | 10.0% | 100,000 | 10,000 | **10,000** |
| Thermalite Core | BUY | +20% | 10.0% | 100,000 | 10,000 | **10,000** |
| Scoria Paste | BUY | +15% | 7.5% | 75,000 | 5,625 | **5,625** |
| Volcanic Incense | BUY | +15% | 7.5% | 75,000 | 5,625 | **5,625** |
| Ashes of Phoenix | SELL | −10% | 5.0% | 50,000 | 2,500 | **2,500** |
| **TOTAL** | | | **100.0%** | **1,000,000** | **127,500** | **127,500** |

**Expected net PnL: 127,500 XIRECs (12.75% ROI on budget)**

---

## Signal Confidence Ranking

| Rank | Product | Catalyst Type | Confidence |
|---|---|---|---|
| 1 | Lava Cake SELL | Regulatory ban + lawsuits + sales halt | VERY HIGH |
| 2 | Obsidian Cutlery BUY | Full manufacturing halt | HIGH |
| 3 | Pyroflex Cells SELL | Tax shock effective tomorrow | HIGH |
| 4 | Magma Ink BUY | Hot drop + merger | HIGH |
| 5 | Sulfur Reactor BUY | Index inclusion (forced buying) | HIGH |
| 6 | Thermalite Core BUY | 3× user growth forecast | MODERATE-HIGH |
| 7 | Scoria Paste BUY | Influencer + infrastructure staple | MODERATE |
| 8 | Volcanic Incense BUY | Ongoing rally + influencer amplification | MODERATE |
| 9 | Ashes of Phoenix SELL | PR controversy only | LOW |

---

## Sensitivity Analysis

What if all expected returns are scaled by a factor X (e.g., if reality is only
half as dramatic as the news implies)?

| Scale | Total Alloc | Expected Net PnL | ROI |
|---|---|---|---|
| 0.25× (very muted) | 25.0% | 7,969 | 0.8% |
| 0.50× (half returns) | 50.0% | 31,875 | 3.2% |
| 0.75× (mild dampening) | 75.0% | 71,719 | 7.2% |
| **1.00× (base case)** | **100.0%** | **127,500** | **12.75%** |
| 1.25× (stronger moves) | 100.0% | 191,016 | 19.1% |

Note: at 1.25× scale the unconstrained total exceeds 100%, so allocations are
scaled down proportionally, losing some optimal compounding.

---

## Key Risks

| Risk | Products Affected | Mitigation |
|---|---|---|
| Lava Cake review resolves quickly | Lava Cake | Accept: hard catalyst, day-1 impact guaranteed |
| Pyroflex cut gets reversed | Pyroflex Cells | Low probability — decision already final |
| Influencer trades reverse | Scoria Paste, Volcanic Incense | Both have secondary fundamentals as backstop |
| Ashes of Phoenix rebounds | Ashes of Phoenix | Low conviction; only 5% allocation |
| Thermalite forecast overstated | Thermalite Core | Conservative 20% estimate; 10% allocation |
| Obsidian supply restores quickly | Obsidian Cutlery | Manufacturing halt takes days to weeks to resolve |

---

## Submission Values

```
Lava Cake          → SELL  20.0%
Obsidian Cutlery   → BUY   15.0%
Pyroflex Cells     → SELL  12.5%
Magma Ink          → BUY   12.5%
Sulfur Reactor     → BUY   10.0%
Thermalite Core    → BUY   10.0%
Scoria Paste       → BUY    7.5%
Volcanic Incense   → BUY    7.5%
Ashes of Phoenix   → SELL   5.0%

Total: 100.0%
Expected net PnL: 127,500 XIRECs
```
