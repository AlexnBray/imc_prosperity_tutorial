import { useApp } from '../store.js';
import type { ParsedLog } from '../../../preload/index.js';

interface Props {
  log: ParsedLog;
}

const AXES = ['y', 'y2', 'y3'] as const;

export function SignalToggles({ log }: Props) {
  const { visibleSignals, axisAssignments, toggleSignal, setAxis } = useApp();

  const all = [
    'mid_price',
    ...log.signalColumns,
    ...(['bid_price_1', 'bid_price_2', 'bid_price_3', 'ask_price_1', 'ask_price_2', 'ask_price_3',
         'algo_bid_price_1', 'algo_bid_price_2', 'algo_bid_price_3',
         'algo_ask_price_1', 'algo_ask_price_2', 'algo_ask_price_3']),
  ];

  const groups: Record<string, string[]> = {
    'Signals': ['mid_price', ...log.signalColumns],
    'Mkt Depth': ['bid_price_1', 'bid_price_2', 'bid_price_3', 'ask_price_1', 'ask_price_2', 'ask_price_3'],
    'Algo Book': ['algo_bid_price_1', 'algo_bid_price_2', 'algo_bid_price_3',
                  'algo_ask_price_1', 'algo_ask_price_2', 'algo_ask_price_3'],
  };

  return (
    <div>
      <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.06em', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 8 }}>
        Chart Layers
      </div>
      {Object.entries(groups).map(([groupName, keys]) => (
        <div key={groupName} style={{ marginBottom: 10 }}>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 4 }}>{groupName}</div>
          {keys.map((k) => {
            const isVisible = visibleSignals.has(k);
            const axis = axisAssignments[k] ?? 'y';
            return (
              <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
                <button
                  onClick={() => toggleSignal(k)}
                  style={{
                    flex: 1,
                    fontSize: 10,
                    padding: '2px 6px',
                    borderRadius: 4,
                    border: '1px solid var(--border)',
                    background: isVisible ? 'var(--accent)' : 'transparent',
                    color: isVisible ? '#fff' : 'var(--text-muted)',
                    cursor: 'pointer',
                    textAlign: 'left',
                    fontFamily: 'monospace',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  }}
                  title={k}
                >
                  {k}
                </button>
                <select
                  value={axis}
                  onChange={(e) => setAxis(k, e.target.value as 'y' | 'y2' | 'y3')}
                  style={{
                    fontSize: 10,
                    padding: '2px 4px',
                    background: 'var(--surface)',
                    color: 'var(--text)',
                    border: '1px solid var(--border)',
                    borderRadius: 4,
                    cursor: 'pointer',
                  }}
                >
                  {AXES.map((a) => <option key={a} value={a}>{a}</option>)}
                </select>
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}
