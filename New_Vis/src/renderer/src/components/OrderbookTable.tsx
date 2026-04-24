import { useMemo } from 'react';
import type { ProductData } from '../../../preload/index.js';
import { useApp } from '../store.js';

interface Props {
  data: ProductData;
}

interface BookRow {
  level: number;
  bidVol: number | null;
  bidPx: number | null;
  askPx: number | null;
  askVol: number | null;
}

function findRowIndex(timestamps: number[], ts: number): number {
  // Binary search for nearest timestamp
  let lo = 0, hi = timestamps.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (timestamps[mid] < ts) lo = mid + 1;
    else hi = mid;
  }
  // Check lo-1 vs lo for nearest
  if (lo > 0 && Math.abs(timestamps[lo - 1] - ts) < Math.abs(timestamps[lo] - ts)) {
    return lo - 1;
  }
  return lo;
}

export function OrderbookTable({ data }: Props) {
  const { activeTimestamp } = useApp();
  const cols = data.columns;
  const ts = data.timestamps;

  const { mktRows, algoRows, recentFills } = useMemo(() => {
    if (activeTimestamp === null || ts.length === 0) {
      return { mktRows: [], algoRows: [], recentFills: [] };
    }

    const idx = findRowIndex(ts, activeTimestamp);

    const mktRows: BookRow[] = [1, 2, 3].map((lvl) => ({
      level: lvl,
      bidVol: cols[`bid_volume_${lvl}`]?.[idx] || null,
      bidPx: cols[`bid_price_${lvl}`]?.[idx] || null,
      askPx: cols[`ask_price_${lvl}`]?.[idx] || null,
      askVol: cols[`ask_volume_${lvl}`]?.[idx] || null,
    }));

    const algoRows: BookRow[] = [1, 2, 3].map((lvl) => ({
      level: lvl,
      bidVol: cols[`algo_bid_volume_${lvl}`]?.[idx] || null,
      bidPx: cols[`algo_bid_price_${lvl}`]?.[idx] || null,
      askPx: cols[`algo_ask_price_${lvl}`]?.[idx] || null,
      askVol: cols[`algo_ask_volume_${lvl}`]?.[idx] || null,
    }));

    // Collect recent fills within 40 rows back
    const fills: Array<{ ts: number; side: string; qty: number; price: number }> = [];
    const lookback = Math.max(0, idx - 40);
    for (let i = idx; i >= lookback; i--) {
      for (let lvl = 1; lvl <= 3; lvl++) {
        const av = cols[`algo_ask_fill_${lvl}_volume`]?.[i] ?? 0;
        const ap = cols[`algo_ask_fill_${lvl}_price`]?.[i] ?? 0;
        if (av > 0) fills.push({ ts: ts[i], side: 'Buy', qty: av, price: ap });
        const bv = cols[`algo_bid_fill_${lvl}_volume`]?.[i] ?? 0;
        const bp = cols[`algo_bid_fill_${lvl}_price`]?.[i] ?? 0;
        if (bv > 0) fills.push({ ts: ts[i], side: 'Sell', qty: bv, price: bp });
      }
    }

    return { mktRows, algoRows, recentFills: fills.slice(0, 8) };
  }, [activeTimestamp, data]);

  const thStyle: React.CSSProperties = {
    fontSize: 10,
    fontWeight: 600,
    padding: '3px 6px',
    textAlign: 'right',
    whiteSpace: 'nowrap',
    borderBottom: '1px solid var(--border)',
  };
  const tdStyle: React.CSSProperties = {
    fontSize: 11,
    padding: '2px 6px',
    textAlign: 'right',
    fontFamily: 'monospace',
  };

  const BookTable = ({ rows, title }: { rows: BookRow[]; title: string }) => (
    <div style={{ marginBottom: 12 }}>
      <p style={{ fontSize: 11, fontWeight: 600, margin: '0 0 4px 0', color: 'var(--text-muted)' }}>
        {title}
      </p>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th style={{ ...thStyle, textAlign: 'left' }}>Lvl</th>
            <th style={{ ...thStyle, color: '#059669' }}>Bid Vol</th>
            <th style={{ ...thStyle, color: '#059669' }}>Bid Px</th>
            <th style={{ ...thStyle, color: '#dc2626' }}>Ask Px</th>
            <th style={{ ...thStyle, color: '#dc2626' }}>Ask Vol</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.level}>
              <td style={{ ...tdStyle, textAlign: 'left', color: 'var(--text-muted)' }}>{r.level}</td>
              <td style={{ ...tdStyle, color: '#059669' }}>{r.bidVol ?? '—'}</td>
              <td style={{ ...tdStyle, color: '#059669', fontWeight: 600 }}>{r.bidPx?.toFixed(1) ?? '—'}</td>
              <td style={{ ...tdStyle, color: '#dc2626', fontWeight: 600 }}>{r.askPx?.toFixed(1) ?? '—'}</td>
              <td style={{ ...tdStyle, color: '#dc2626' }}>{r.askVol ?? '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );

  return (
    <div style={{ fontSize: 12 }}>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, fontFamily: 'monospace' }}>
        T = {activeTimestamp ?? '—'}
      </div>

      <BookTable rows={mktRows} title="Market Depth" />
      <BookTable rows={algoRows} title="Algo Quotes" />

      <p style={{ fontSize: 11, fontWeight: 600, margin: '0 0 4px 0', color: 'var(--text-muted)' }}>
        Recent Fills
      </p>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th style={{ ...thStyle, textAlign: 'left' }}>Time</th>
            <th style={thStyle}>Side</th>
            <th style={thStyle}>Px</th>
            <th style={thStyle}>Qty</th>
          </tr>
        </thead>
        <tbody>
          {recentFills.length === 0 ? (
            <tr><td colSpan={4} style={{ ...tdStyle, textAlign: 'center', color: 'var(--text-muted)' }}>—</td></tr>
          ) : (
            recentFills.map((f, i) => (
              <tr key={i}>
                <td style={{ ...tdStyle, textAlign: 'left', color: 'var(--text-muted)' }}>{f.ts}</td>
                <td style={{ ...tdStyle, color: f.side === 'Buy' ? '#059669' : '#dc2626' }}>{f.side}</td>
                <td style={{ ...tdStyle }}>{f.price.toFixed(1)}</td>
                <td style={{ ...tdStyle }}>{f.qty}</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
