import { useApp } from '../store.js';
import type { ParsedLog } from '../../../preload/index.js';

interface Props {
  log: ParsedLog | null;
  onOpen: () => void;
  onThemeToggle: () => void;
}

export function Header({ log, onOpen, onThemeToggle }: Props) {
  const { activeProduct, setActiveProduct, prefs } = useApp();
  const isDark = prefs?.theme === 'dark';

  return (
    <header
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        padding: '8px 16px',
        borderBottom: '1px solid var(--border)',
        background: 'var(--surface)',
        flexShrink: 0,
      }}
    >
      <span style={{ fontFamily: 'monospace', fontWeight: 700, fontSize: 14, whiteSpace: 'nowrap', color: 'var(--text)' }}>
        Prosperity Vis
      </span>

      <button className="btn" onClick={onOpen} title="Open log file">
        Open Log
      </button>

      {log && (
        <>
          <span style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'monospace', maxWidth: 260, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {log.name}
          </span>

          <select
            value={activeProduct ?? ''}
            onChange={(e) => setActiveProduct(e.target.value)}
            className="select"
          >
            {log.products.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
        </>
      )}

      <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', fontSize: 12, color: 'var(--text-muted)' }}>
          <input
            type="checkbox"
            checked={isDark}
            onChange={onThemeToggle}
            style={{ cursor: 'pointer' }}
          />
          Dark
        </label>
      </div>
    </header>
  );
}
