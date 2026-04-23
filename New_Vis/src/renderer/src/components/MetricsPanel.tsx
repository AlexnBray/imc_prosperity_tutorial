import type { ProductData, Trade } from '../../../preload/index.js';
import { useApp } from '../store.js';
import { METRICS, METRIC_BY_KEY } from '../metrics/registry.js';

interface Props {
  data: ProductData;
  trades: Trade[];
}

export function MetricsPanel({ data, trades }: Props) {
  const { prefs, toggleMetric } = useApp();
  const enabledKeys = prefs?.enabledMetrics ?? [];

  const ctx = { data, trades, product: data.product };

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
        <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.06em', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
          Performance
        </span>
      </div>

      {/* Toggleable metric chips */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 10 }}>
        {METRICS.map((m) => {
          const active = enabledKeys.includes(m.key);
          return (
            <button
              key={m.key}
              onClick={() => toggleMetric(m.key)}
              style={{
                fontSize: 9,
                padding: '2px 6px',
                borderRadius: 4,
                border: '1px solid var(--border)',
                background: active ? 'var(--accent)' : 'transparent',
                color: active ? '#fff' : 'var(--text-muted)',
                cursor: 'pointer',
                fontFamily: 'monospace',
              }}
            >
              {m.label}
            </button>
          );
        })}
      </div>

      {/* Active metric cards */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
        {enabledKeys.map((key) => {
          const def = METRIC_BY_KEY.get(key);
          if (!def) return null;
          const value = def.compute(ctx);
          const colorClass = def.signColor?.(value);
          const color =
            colorClass === 'pos' ? '#10b981' :
            colorClass === 'neg' ? '#f43f5e' :
            'var(--text)';
          return (
            <div
              key={key}
              style={{
                background: 'var(--surface)',
                border: '1px solid var(--border)',
                borderRadius: 6,
                padding: '6px 8px',
              }}
            >
              <div style={{ fontSize: 9, color: 'var(--text-muted)', marginBottom: 2 }}>{def.label}</div>
              <div style={{ fontSize: 14, fontFamily: 'monospace', color, fontWeight: 600 }}>
                {def.format(value)}
              </div>
              {def.subtitle && (
                <div style={{ fontSize: 9, color: 'var(--text-muted)' }}>{def.subtitle}</div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
