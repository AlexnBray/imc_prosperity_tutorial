
    

# ── Position limits ──────────────────────────────────────────────────────────

LIMITS: dict[Symbol, int] = {
    "EMERALDS": 50,
    "TOMATOES": 50,
}

# Emeralds: stationary ~10,000 fair value
EMERALDS_FAIR_VALUE = 10_000
EMERALDS_SPREAD     = 7
EMERALDS_TAKE_EDGE  = 0

# Tomatoes: drifting — regime detected via Kalman filter
TOMATOES_TAKE_EDGE  = 1
KALMAN_Q = 0.017  
KALMAN_R = 0.245  

VELOCITY_THRESHOLD = 0.15


# ── Utility functions ────────────────────────────────────────────────────────

def best_bid(od: OrderDepth) -> tuple[int, int] | None:
    if not od.buy_orders:
        return None
    p = max(od.buy_orders)
    return p, od.buy_orders[p]


def best_ask(od: OrderDepth) -> tuple[int, int] | None:
    if not od.sell_orders:
        return None
    p = min(od.sell_orders)
    return p, od.sell_orders[p]


def mid_price(od: OrderDepth) -> float | None:
    b = best_bid(od)
    a = best_ask(od)
    if b and a:
        return (b[0] + a[0]) / 2.0
    return None


def buy_capacity(position: int, limit: int) -> int:
    return limit - position


def sell_capacity(position: int, limit: int) -> int:
    return limit + position


# ── Kalman Filter ─────────────────────────────────────────────────────────────

class KalmanFilter:
    """
    Tracks the true price and its velocity (trend direction/strength).

    Two steps every tick:
      Predict: extrapolate price forward using current velocity
      Update:  blend prediction with new observation weighted by Kalman gain K

    Outputs:
      price    — smoothed estimate of the true mid price
      velocity — rate of change (+ve = uptrend, -ve = downtrend)
    """

    def __init__(self, process_noise: float = KALMAN_Q, observation_noise: float = KALMAN_R) -> None:
        self.price    : float | None = None
        self.velocity : float        = 0.0
        self.P        : float        = 1.0   # estimate uncertainty

        self.Q = process_noise      # expected process noise per tick
        self.R = observation_noise  # expected observation noise

    def update(self, observed: float) -> tuple[float, float]:
        """Feed in a new mid price. Returns (estimated_price, velocity)."""
        # First tick — just initialise, no prediction yet
        if self.price is None:
            self.price = observed
            return self.price, self.velocity

        # ── Predict ───────────────────────────────────────────────────────
        predicted_price = self.price + self.velocity
        predicted_P     = self.P + self.Q

        # ── Update ────────────────────────────────────────────────────────
        K             = predicted_P / (predicted_P + self.R)   # Kalman gain
        error         = observed - predicted_price

        self.price    = predicted_price + K * error
        self.velocity = self.velocity   + K * error * 0.1      # gentle velocity correction
        self.P        = (1 - K) * predicted_P

        return self.price, self.velocity

    def regime(self) -> int:
        """
        Classify current trend direction from velocity:
          +1 = uptrend   (buy bias — tighten ask, widen bid)
          -1 = downtrend (sell bias — tighten bid, widen ask)
           0 = flat/unknown
        """
        if self.velocity >  VELOCITY_THRESHOLD:
            return  1
        if self.velocity < -VELOCITY_THRESHOLD:
            return -1
        return 0

    def to_dict(self) -> dict:
        return {
            "price":    self.price,
            "velocity": self.velocity,
            "P":        self.P,
        }

    @staticmethod
    def from_dict(d: dict) -> "KalmanFilter":
        kf = KalmanFilter()
        kf.price    = d.get("price")
        kf.velocity = d.get("velocity", 0.0)
        kf.P        = d.get("P", 1.0)
        return kf


# ── Emeralds strategy ─────────────────────────────────────────────────────────

def strategy_emeralds(od: OrderDepth, position: int, limit: int) -> list[Order]:
    """
    Stationary fair value of 10,000.
    1. Aggressively take any mis-priced orders crossing fair.
    2. Post passive quotes EMERALDS_SPREAD ticks either side.
    """
    orders: list[Order] = []
    fv  = EMERALDS_FAIR_VALUE
    pos = position

    # Aggressive takes
    for ask in sorted(od.sell_orders):
        if ask >= fv - EMERALDS_TAKE_EDGE:
            break
        qty = min(abs(od.sell_orders[ask]), buy_capacity(pos, limit))
        if qty <= 0:
            break
        orders.append(Order("EMERALDS", ask, qty))
        pos += qty

    for bid in sorted(od.buy_orders, reverse=True):
        if bid <= fv + EMERALDS_TAKE_EDGE:
            break
        qty = min(od.buy_orders[bid], sell_capacity(pos, limit))
        if qty <= 0:
            break
        orders.append(Order("EMERALDS", bid, -qty))
        pos -= qty

    # Passive quotes
    buy_qty  = min(buy_capacity(pos, limit),  10)
    sell_qty = min(sell_capacity(pos, limit), 10)

    if buy_qty  > 0:
        orders.append(Order("EMERALDS", fv - EMERALDS_SPREAD,  buy_qty))
    if sell_qty > 0:
        orders.append(Order("EMERALDS", fv + EMERALDS_SPREAD, -sell_qty))

    return orders


# ── Tomatoes strategy ─────────────────────────────────────────────────────────

def strategy_tomatoes(
    od      : OrderDepth,
    position: int,
    limit   : int,
    kf      : KalmanFilter,
) -> list[Order]:
    """
    Uses the Kalman filter for both fair value estimation and regime detection.

    Regime skews passive quote spreads:
      Uptrend   (+1): tighter ask (easier to sell into rally), wider bid
      Downtrend (-1): tighter bid (easier to buy the dip),    wider ask
      Flat       (0): symmetric quotes
    """
    orders: list[Order] = []

    fair = kf.price if kf.price is not None else mid_price(od)
    if fair is None:
        return orders

    r = kf.regime()

    if r == 1:        # uptrend — lean long
        bid_spread, ask_spread = 2, 4
    elif r == -1:     # downtrend — lean short
        bid_spread, ask_spread = 4, 2
    else:             # flat / warming up
        bid_spread, ask_spread = 3, 3

    pos = position

    # Aggressive takes against Kalman fair value
    for ask in sorted(od.sell_orders):
        if ask >= fair - TOMATOES_TAKE_EDGE:
            break
        qty = min(abs(od.sell_orders[ask]), buy_capacity(pos, limit))
        if qty <= 0:
            break
        orders.append(Order("TOMATOES", ask, qty))
        pos += qty

    for bid in sorted(od.buy_orders, reverse=True):
        if bid <= fair + TOMATOES_TAKE_EDGE:
            break
        qty = min(od.buy_orders[bid], sell_capacity(pos, limit))
        if qty <= 0:
            break
        orders.append(Order("TOMATOES", bid, -qty))
        pos -= qty

    # Regime-skewed passive quotes
    buy_qty  = min(buy_capacity(pos, limit),  8)
    sell_qty = min(sell_capacity(pos, limit), 8)

    if buy_qty  > 0:
        orders.append(Order("TOMATOES", math.floor(fair) - bid_spread,  buy_qty))
    if sell_qty > 0:
        orders.append(Order("TOMATOES", math.ceil(fair)  + ask_spread, -sell_qty))

    return orders


# ── Persistent trader state ───────────────────────────────────────────────────

class TraderState:
    """
    Persists price history and Kalman filter state across ticks via traderData.
    """

    def __init__(self) -> None:
        self.price_history: dict[str, list[float]] = {}
        self.kalman_states: dict[str, dict]        = {}

    def get_kalman(self, symbol: str) -> KalmanFilter:
        if symbol in self.kalman_states:
            return KalmanFilter.from_dict(self.kalman_states[symbol])
        return KalmanFilter()

    def save_kalman(self, symbol: str, kf: KalmanFilter) -> None:
        self.kalman_states[symbol] = kf.to_dict()

    @staticmethod
    def from_json(raw: str) -> "TraderState":
        ts = TraderState()
        if raw:
            try:
                data = json.loads(raw)
                ts.price_history = data.get("price_history", {})
                ts.kalman_states = data.get("kalman_states", {})
            except Exception:
                pass
        return ts

    def to_json(self) -> str:
        return json.dumps({
            "price_history": self.price_history,
            "kalman_states": self.kalman_states,
        })


# ── Main Trader class ─────────────────────────────────────────────────────────

class Trader:

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        ts = TraderState.from_json(state.traderData)

        result     : dict[Symbol, list[Order]] = {}
        conversions = 0

        for symbol, od in state.order_depths.items():
            limit    = LIMITS.get(symbol, 20)
            position = state.position.get(symbol, 0)
            mp       = mid_price(od)

            if symbol == "EMERALDS":
                orders = strategy_emeralds(od, position, limit)

            elif symbol == "TOMATOES":
                # Load Kalman state, feed new mid price, run strategy, save state
                kf = ts.get_kalman(symbol)
                if mp is not None:
                    kf.update(mp)
                ts.save_kalman(symbol, kf)
                orders = strategy_tomatoes(od, position, limit, kf)

            else:
                # Generic fallback for new products in later rounds
                orders = []
                if mp is not None:
                    spread = 3
                    if buy_capacity(position, limit) > 0:
                        orders.append(Order(symbol, math.floor(mp) - spread, 5))
                    if sell_capacity(position, limit) > 0:
                        orders.append(Order(symbol, math.ceil(mp)  + spread, -5))

            if orders:
                result[symbol] = orders

            # Log per-symbol summary for the visualiser
            kf_price = ts.kalman_states.get(symbol, {}).get("price")
            kf_vel   = ts.kalman_states.get(symbol, {}).get("velocity")
            logger.print(
                f"{symbol:12s}  pos={position:+4d}  mid={mp!s:>8}  "
                f"kf_price={kf_price!s:>8}  kf_vel={kf_vel!s:>7}  orders={len(orders)}"
            )
