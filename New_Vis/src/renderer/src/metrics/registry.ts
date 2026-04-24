import type { ProductData, Trade } from '../../../preload/index.js';

export interface MetricContext {
  data: ProductData;
  trades: Trade[];
  product: string;
}

export interface MetricDefinition {
  key: string;
  label: string;
  subtitle?: string;
  format: (v: number | null) => string;
  compute: (ctx: MetricContext) => number | null;
  signColor?: (v: number | null) => 'pos' | 'neg' | 'neutral';
}

const ADVERSE_WINDOW = 5;

function col(ctx: MetricContext, key: string): number[] {
  return ctx.data.columns[key] ?? [];
}

function diff(a: number[]): number[] {
  const out = new Array(a.length).fill(0);
  for (let i = 1; i < a.length; i++) out[i] = a[i] - a[i - 1];
  return out;
}

function mean(a: number[]): number {
  if (a.length === 0) return 0;
  let s = 0;
  for (const v of a) s += v;
  return s / a.length;
}

function std(a: number[]): number {
  if (a.length < 2) return 0;
  const m = mean(a);
  let s = 0;
  for (const v of a) s += (v - m) ** 2;
  return Math.sqrt(s / (a.length - 1));
}

function fillVolume(ctx: MetricContext, side: 'bid' | 'ask'): number[] {
  const n = ctx.data.timestamps.length;
  const out = new Array(n).fill(0);
  for (let lvl = 1; lvl <= 3; lvl++) {
    const v = col(ctx, `algo_${side}_fill_${lvl}_volume`);
    for (let i = 0; i < n; i++) out[i] += v[i] ?? 0;
  }
  return out;
}

const fmtFixed = (digits: number) => (v: number | null) =>
  v === null || !Number.isFinite(v) ? 'N/A' : v.toFixed(digits);
const fmtSigned = (digits: number) => (v: number | null) =>
  v === null || !Number.isFinite(v) ? 'N/A' : (v >= 0 ? '+' : '') + v.toFixed(digits);
const fmtInt = (v: number | null) =>
  v === null || !Number.isFinite(v) ? 'N/A' : Math.round(v).toString();
const fmtPct = (v: number | null) =>
  v === null || !Number.isFinite(v) ? 'N/A' : v.toFixed(2) + '%';

const signByValue = (v: number | null): 'pos' | 'neg' | 'neutral' =>
  v === null || !Number.isFinite(v) ? 'neutral' : v > 0 ? 'pos' : v < 0 ? 'neg' : 'neutral';

export const METRICS: MetricDefinition[] = [
  {
    key: 'final_pnl',
    label: 'Final PnL',
    subtitle: 'last PnL value',
    format: fmtSigned(2),
    signColor: signByValue,
    compute: (ctx) => {
      const p = col(ctx, 'profit_and_loss');
      return p.length ? p[p.length - 1] : null;
    }
  },
  {
    key: 'sharpe',
    label: 'Sharpe',
    subtitle: 'episode-scaled',
    format: fmtFixed(3),
    signColor: (v) => (v === null ? 'neutral' : v >= 1 ? 'pos' : v >= 0 ? 'neutral' : 'neg'),
    compute: (ctx) => {
      const d = diff(col(ctx, 'profit_and_loss'));
      const s = std(d);
      return s > 0 ? (mean(d) / s) * Math.sqrt(d.length) : 0;
    }
  },
  {
    key: 'max_drawdown',
    label: 'Max Drawdown',
    subtitle: 'peak → trough',
    format: fmtSigned(2),
    signColor: () => 'neg',
    compute: (ctx) => {
      const p = col(ctx, 'profit_and_loss');
      let peak = -Infinity;
      let minDd = 0;
      for (const v of p) {
        if (v > peak) peak = v;
        const dd = v - peak;
        if (dd < minDd) minDd = dd;
      }
      return minDd;
    }
  },
  {
    key: 'volatility',
    label: 'PnL Volatility',
    subtitle: 'std of PnL Δ',
    format: fmtFixed(3),
    compute: (ctx) => std(diff(col(ctx, 'profit_and_loss')))
  },
  {
    key: 'position_skew',
    label: 'Position Skew',
    subtitle: 'time-avg signed pos',
    format: fmtSigned(2),
    signColor: (v) => (v === null ? 'neutral' : Math.abs(v) > 5 ? 'neg' : 'pos'),
    compute: (ctx) => mean(col(ctx, 'position'))
  },
  {
    key: 'fill_rate',
    label: 'Fill Rate',
    subtitle: '% ticks with any fill',
    format: fmtPct,
    compute: (ctx) => {
      const bids = fillVolume(ctx, 'bid');
      const asks = fillVolume(ctx, 'ask');
      const n = bids.length;
      if (n === 0) return 0;
      let active = 0;
      for (let i = 0; i < n; i++) if (bids[i] + asks[i] > 0) active++;
      return (active / n) * 100;
    }
  },
  {
    key: 'total_volume',
    label: 'Total Volume',
    subtitle: 'algo buys + sells',
    format: fmtInt,
    compute: (ctx) => {
      const b = fillVolume(ctx, 'ask'); // we buy at the ask
      const s = fillVolume(ctx, 'bid'); // we sell at the bid
      return b.reduce((a, x) => a + x, 0) + s.reduce((a, x) => a + x, 0);
    }
  },
  {
    key: 'buy_edge',
    label: 'Buy Edge',
    subtitle: 'mid − avg buy px',
    format: fmtFixed(3),
    signColor: signByValue,
    compute: (ctx) => {
      const vol = fillVolume(ctx, 'ask');
      const mid = col(ctx, 'mid_price');
      let vwap = 0, totVol = 0, midSum = 0, midN = 0;
      for (let lvl = 1; lvl <= 3; lvl++) {
        const v = col(ctx, `algo_ask_fill_${lvl}_volume`);
        const p = col(ctx, `algo_ask_fill_${lvl}_price`);
        for (let i = 0; i < v.length; i++) {
          if (v[i] > 0) vwap += p[i] * v[i];
        }
      }
      for (let i = 0; i < vol.length; i++) {
        if (vol[i] > 0) {
          totVol += vol[i];
          midSum += mid[i];
          midN++;
        }
      }
      if (totVol === 0 || midN === 0) return null;
      return midSum / midN - vwap / totVol;
    }
  },
  {
    key: 'sell_edge',
    label: 'Sell Edge',
    subtitle: 'avg sell px − mid',
    format: fmtFixed(3),
    signColor: signByValue,
    compute: (ctx) => {
      const vol = fillVolume(ctx, 'bid');
      const mid = col(ctx, 'mid_price');
      let vwap = 0, totVol = 0, midSum = 0, midN = 0;
      for (let lvl = 1; lvl <= 3; lvl++) {
        const v = col(ctx, `algo_bid_fill_${lvl}_volume`);
        const p = col(ctx, `algo_bid_fill_${lvl}_price`);
        for (let i = 0; i < v.length; i++) {
          if (v[i] > 0) vwap += p[i] * v[i];
        }
      }
      for (let i = 0; i < vol.length; i++) {
        if (vol[i] > 0) {
          totVol += vol[i];
          midSum += mid[i];
          midN++;
        }
      }
      if (totVol === 0 || midN === 0) return null;
      return vwap / totVol - midSum / midN;
    }
  },
  {
    key: 'adverse_buy',
    label: `Adv Sel Buy (${ADVERSE_WINDOW}t)`,
    subtitle: 'mid Δ after buy',
    format: fmtFixed(3),
    signColor: (v) => (v === null ? 'neutral' : v >= 0 ? 'pos' : 'neg'),
    compute: (ctx) => {
      const mid = col(ctx, 'mid_price');
      const buys = fillVolume(ctx, 'ask');
      const out: number[] = [];
      for (let i = 0; i + ADVERSE_WINDOW < mid.length; i++) {
        if (buys[i] > 0) out.push(mid[i + ADVERSE_WINDOW] - mid[i]);
      }
      return out.length ? mean(out) : null;
    }
  },
  {
    key: 'adverse_sell',
    label: `Adv Sel Sell (${ADVERSE_WINDOW}t)`,
    subtitle: 'mid Δ after sell',
    format: fmtFixed(3),
    signColor: (v) => (v === null ? 'neutral' : v <= 0 ? 'pos' : 'neg'),
    compute: (ctx) => {
      const mid = col(ctx, 'mid_price');
      const sells = fillVolume(ctx, 'bid');
      const out: number[] = [];
      for (let i = 0; i + ADVERSE_WINDOW < mid.length; i++) {
        if (sells[i] > 0) out.push(mid[i + ADVERSE_WINDOW] - mid[i]);
      }
      return out.length ? mean(out) : null;
    }
  }
];

export const METRIC_BY_KEY = new Map(METRICS.map((m) => [m.key, m]));
