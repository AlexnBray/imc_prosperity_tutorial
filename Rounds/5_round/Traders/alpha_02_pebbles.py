# Alpha 02 — PEBBLES cross-family signal targets
# Local 3-day PnL: ~$157,787
#
# Signals:
#   FAM_SNACKPACK (family factor)        -> PEBBLES_XL  direction=-1
#   REL_GALAXY_SOUNDS_SOLAR_FLAMES       -> PEBBLES_XL  direction=+1
#   REL_GALAXY_SOUNDS_BLACK_HOLES        -> PEBBLES_XS  direction=+1
#   REL_TRANSLATOR_GRAPHITE_MIST         -> PEBBLES_M   direction=+1
#   REL_SLEEP_POD_POLYESTER              -> PEBBLES_L   direction=-1
#
# Do NOT add same-family PEBBLES arb — Hurst ~0.95, they trend not revert.

import json
from datamodel import Order, TradingState

try:
    import prosperity4bt.data as _btd
    for _p in [
        "GALAXY_SOUNDS_BLACK_HOLES","GALAXY_SOUNDS_DARK_MATTER","GALAXY_SOUNDS_PLANETARY_RINGS","GALAXY_SOUNDS_SOLAR_FLAMES","GALAXY_SOUNDS_SOLAR_WINDS",
        "PEBBLES_L","PEBBLES_M","PEBBLES_S","PEBBLES_XL","PEBBLES_XS",
        "SLEEP_POD_COTTON","SLEEP_POD_LAMB_WOOL","SLEEP_POD_NYLON","SLEEP_POD_POLYESTER","SLEEP_POD_SUEDE",
        "SNACKPACK_CHOCOLATE","SNACKPACK_PISTACHIO","SNACKPACK_RASPBERRY","SNACKPACK_STRAWBERRY","SNACKPACK_VANILLA",
        "TRANSLATOR_ASTRO_BLACK","TRANSLATOR_ECLIPSE_CHARCOAL","TRANSLATOR_GRAPHITE_MIST","TRANSLATOR_SPACE_GRAY","TRANSLATOR_VOID_BLUE",
    ]:
        _btd.LIMITS.setdefault(_p, 10)
    del _btd, _p
except ImportError:
    pass

FAMILIES = {
    "GALAXY_SOUNDS": ["GALAXY_SOUNDS_BLACK_HOLES","GALAXY_SOUNDS_DARK_MATTER","GALAXY_SOUNDS_PLANETARY_RINGS","GALAXY_SOUNDS_SOLAR_FLAMES","GALAXY_SOUNDS_SOLAR_WINDS"],
    "SLEEP_POD":     ["SLEEP_POD_COTTON","SLEEP_POD_LAMB_WOOL","SLEEP_POD_NYLON","SLEEP_POD_POLYESTER","SLEEP_POD_SUEDE"],
    "SNACKPACK":     ["SNACKPACK_CHOCOLATE","SNACKPACK_PISTACHIO","SNACKPACK_RASPBERRY","SNACKPACK_STRAWBERRY","SNACKPACK_VANILLA"],
    "TRANSLATOR":    ["TRANSLATOR_ASTRO_BLACK","TRANSLATOR_ECLIPSE_CHARCOAL","TRANSLATOR_GRAPHITE_MIST","TRANSLATOR_SPACE_GRAY","TRANSLATOR_VOID_BLUE"],
}

SIGNALS = [
    {"key":"fam_snackpack_to_peb_xl",            "kind":"family_factor",   "family":"SNACKPACK",                           "target":"PEBBLES_XL", "lookback":50,  "threshold":2.972055365611617e-05,   "hold":50, "direction_sign":-1, "qty":10},
    {"key":"gs_sf_to_peb_xl",                    "kind":"family_relative", "source":"GALAXY_SOUNDS_SOLAR_FLAMES",           "target":"PEBBLES_XL", "lookback":20,  "threshold":0.0001563329859531438,   "hold":50, "direction_sign": 1, "qty":10},
    {"key":"gs_bh_to_peb_xs",                    "kind":"family_relative", "source":"GALAXY_SOUNDS_BLACK_HOLES",            "target":"PEBBLES_XS", "lookback":50,  "threshold":0.0001406658066183674,   "hold":50, "direction_sign": 1, "qty":10},
    {"key":"translator_graphite_mist_to_peb_m",  "kind":"family_relative", "source":"TRANSLATOR_GRAPHITE_MIST",             "target":"PEBBLES_M",  "lookback":200, "threshold":8.103776922545124e-05,   "hold":50, "direction_sign": 1, "qty":10},
    {"key":"sleep_polyester_to_peb_l",           "kind":"family_relative", "source":"SLEEP_POD_POLYESTER",                  "target":"PEBBLES_L",  "lookback":500, "threshold":6.519165112824712e-05,   "hold":50, "direction_sign":-1, "qty":10},
]

FLATTEN_TS = 990000

# --- derived constants ---
PRODUCT_TO_FAMILY = {p: f for f, members in FAMILIES.items() for p in members}
REL_SOURCES = sorted({s["source"] for s in SIGNALS if s["kind"] == "family_relative"})
FACTOR_FAMILIES = sorted({s["family"] for s in SIGNALS if s["kind"] == "family_factor"})
SOURCE_TO_SIBLINGS = {src: [p for p in FAMILIES[PRODUCT_TO_FAMILY[src]] if p != src] for src in REL_SOURCES}
SOURCE_TO_MAX_LOOKBACK = {}
for sig in SIGNALS:
    if sig["kind"] == "family_relative":
        src = sig["source"]
        SOURCE_TO_MAX_LOOKBACK[src] = max(SOURCE_TO_MAX_LOOKBACK.get(src, 0), sig["lookback"])
FAMILY_TO_MAX_LOOKBACK = {}
for sig in SIGNALS:
    if sig["kind"] == "family_factor":
        f = sig["family"]
        FAMILY_TO_MAX_LOOKBACK[f] = max(FAMILY_TO_MAX_LOOKBACK.get(f, 0), sig["lookback"])
TARGET_PRODUCTS = sorted({s["target"] for s in SIGNALS})
NEEDED = set(TARGET_PRODUCTS)
for src in REL_SOURCES:
    NEEDED.add(src)
    NEEDED.update(SOURCE_TO_SIBLINGS[src])
for f in FACTOR_FAMILIES:
    NEEDED.update(FAMILIES[f])


def best_bid_ask(depth):
    bid = max(depth.buy_orders) if depth.buy_orders else None
    ask = min(depth.sell_orders) if depth.sell_orders else None
    return bid, ask


def best_volume(depth, side):
    if side == "buy":
        return abs(depth.sell_orders[min(depth.sell_orders)]) if depth.sell_orders else 0
    return abs(depth.buy_orders[max(depth.buy_orders)]) if depth.buy_orders else 0


def place_to_target(product, target, state, out):
    depth = state.order_depths.get(product)
    if depth is None:
        return
    bid, ask = best_bid_ask(depth)
    pos = state.position.get(product, 0)
    diff = int(target - pos)
    if diff > 0 and ask is not None:
        qty = min(diff, best_volume(depth, "buy"))
        if qty > 0:
            out.setdefault(product, []).append(Order(product, ask, qty))
    elif diff < 0 and bid is not None:
        qty = min(-diff, best_volume(depth, "sell"))
        if qty > 0:
            out.setdefault(product, []).append(Order(product, bid, -qty))


class Trader:
    def run(self, state: TradingState):
        try:
            data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            data = {}
        result = {}
        mids = {}
        for p in sorted(NEEDED):
            depth = state.order_depths.get(p)
            if depth is None:
                continue
            bid, ask = best_bid_ask(depth)
            if bid is not None and ask is not None:
                mids[p] = 0.5 * (bid + ask)
        if len(mids) < len(NEEDED):
            return result, 0, json.dumps(data)

        tick = int(data.get("tick", 0)) + 1
        data["tick"] = tick

        rel_state = data.get("rel_state", {})
        for src in REL_SOURCES:
            sk = rel_state.get(src, {})
            last = sk.get("last_mids", {})
            hist = sk.get("sig_hist", [])
            if last:
                sibs = SOURCE_TO_SIBLINGS[src]
                sig = (mids[src] / last[src] - 1.0) - sum(mids[p] / last[p] - 1.0 for p in sibs) / len(sibs)
                hist.append(sig)
                if len(hist) > SOURCE_TO_MAX_LOOKBACK[src] + 10:
                    hist = hist[-(SOURCE_TO_MAX_LOOKBACK[src] + 10):]
            sk["last_mids"] = {p: mids[p] for p in [src] + SOURCE_TO_SIBLINGS[src]}
            sk["sig_hist"] = hist
            rel_state[src] = sk

        factor_state = data.get("factor_state", {})
        for f in FACTOR_FAMILIES:
            sk = factor_state.get(f, {})
            last = sk.get("last_mids", {})
            hist = sk.get("sig_hist", [])
            members = FAMILIES[f]
            if last:
                hist.append(sum(mids[p] / last[p] - 1.0 for p in members) / len(members))
                if len(hist) > FAMILY_TO_MAX_LOOKBACK[f] + 10:
                    hist = hist[-(FAMILY_TO_MAX_LOOKBACK[f] + 10):]
            sk["last_mids"] = {p: mids[p] for p in members}
            sk["sig_hist"] = hist
            factor_state[f] = sk

        signal_pos = data.get("signal_pos", {})
        signal_exit = data.get("signal_exit", {})
        targets = {p: 0 for p in TARGET_PRODUCTS}

        for sig in SIGNALS:
            hist = rel_state[sig["source"]]["sig_hist"] if sig["kind"] == "family_relative" else factor_state[sig["family"]]["sig_hist"]
            cur = sum(hist[-sig["lookback"]:]) / sig["lookback"] if len(hist) >= sig["lookback"] else None
            key = sig["key"]
            pos_state = int(signal_pos.get(key, 0))
            exit_tick = int(signal_exit.get(key, -1))
            if tick >= exit_tick or state.timestamp >= FLATTEN_TS:
                pos_state = 0
            if state.timestamp < FLATTEN_TS and cur is not None and pos_state == 0 and abs(cur) >= sig["threshold"]:
                pos_state = sig["direction_sign"] if cur > 0 else -sig["direction_sign"]
                exit_tick = tick + sig["hold"]
            targets[sig["target"]] += sig["qty"] * pos_state
            signal_pos[key] = pos_state
            signal_exit[key] = exit_tick

        for p, t in targets.items():
            place_to_target(p, max(-10, min(10, t)), state, result)

        data["rel_state"] = rel_state
        data["factor_state"] = factor_state
        data["signal_pos"] = signal_pos
        data["signal_exit"] = signal_exit
        return result, 0, json.dumps(data)
