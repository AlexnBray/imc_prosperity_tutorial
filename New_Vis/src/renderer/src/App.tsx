import { useEffect, useCallback } from 'react';
import { useApp } from './store.js';
import { Header } from './components/Header.js';
import { PriceChart } from './components/PriceChart.js';
import { PnlChart } from './components/PnlChart.js';
import { OrderbookTable } from './components/OrderbookTable.js';
import { MetricsPanel } from './components/MetricsPanel.js';
import { SignalToggles } from './components/SignalToggles.js';

export function App() {
  const { log, activeProduct, prefs, setLog, setPrefs, setTheme } = useApp();

  // Load persisted prefs on mount
  useEffect(() => {
    window.api.getPrefs().then(setPrefs);
  }, []);

  // Apply theme attribute
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', prefs?.theme ?? 'dark');
  }, [prefs?.theme]);

  const handleOpen = useCallback(async () => {
    const parsed = await window.api.openLogDialog();
    if (parsed) setLog(parsed);
  }, []);

  const handleThemeToggle = useCallback(() => {
    setTheme(prefs?.theme === 'dark' ? 'light' : 'dark');
  }, [prefs?.theme]);

  const isDark = prefs?.theme !== 'light';
  const productData = log && activeProduct ? log.perProduct[activeProduct] : null;
  const trades = log ? log.trades.filter((t: { product: string }) => t.product === activeProduct) : [];

  return (
    <div className="app">
      <Header log={log} onOpen={handleOpen} onThemeToggle={handleThemeToggle} />

      {!log || !productData ? (
        <div className="empty-state">
          <h2>Prosperity Visualiser</h2>
          <p>Open a Prosperity <code>.log</code> file to get started.</p>
          <button className="btn" onClick={handleOpen}>Open Log File</button>
        </div>
      ) : (
        <div className="workspace">
          {/* Signal / layer toggles */}
          <aside className="sidebar-left">
            <SignalToggles log={log} />
          </aside>

          {/* Main chart column */}
          <main className="chart-area">
            <div className="chart-price">
              <PriceChart data={productData} trades={trades} isDark={isDark} />
            </div>
            <div className="chart-pnl">
              <PnlChart data={productData} isDark={isDark} />
            </div>
          </main>

          {/* Orderbook + metrics */}
          <aside className="sidebar-right">
            <OrderbookTable data={productData} />
            <div className="divider" />
            <MetricsPanel data={productData} trades={trades} />
          </aside>
        </div>
      )}
    </div>
  );
}
