import { create } from 'zustand';
import type { ParsedLog, Prefs } from '../../preload/index.js';

interface AppState {
  log: ParsedLog | null;
  activeProduct: string | null;
  activeTimestamp: number | null;
  prefs: Prefs | null;

  visibleSignals: Set<string>;
  axisAssignments: Record<string, 'y' | 'y2' | 'y3'>;

  setLog: (log: ParsedLog | null) => void;
  setActiveProduct: (p: string) => void;
  setActiveTimestamp: (t: number | null) => void;
  setPrefs: (p: Prefs) => void;

  toggleSignal: (name: string) => void;
  setSignalVisible: (name: string, visible: boolean) => void;
  setAxis: (name: string, axis: 'y' | 'y2' | 'y3') => void;
  setSignalColor: (name: string, color: string) => void;
  toggleMetric: (key: string) => void;
  setTheme: (t: 'dark' | 'light') => void;
}

export const useApp = create<AppState>((set, get) => ({
  log: null,
  activeProduct: null,
  activeTimestamp: null,
  prefs: null,

  visibleSignals: new Set<string>(),
  axisAssignments: {},

  setLog: (log) => {
    const first = log?.products[0] ?? null;
    const vis = new Set<string>();
    if (log) {
      for (const s of log.signalColumns) vis.add(s);
      vis.add('mid_price');
      // Keep core orderbook layers visible by default so price chart is never empty.
      ['bid_price_1', 'ask_price_1', 'algo_bid_price_1', 'algo_ask_price_1'].forEach((k) => vis.add(k));
    }
    set({ log, activeProduct: first, activeTimestamp: null, visibleSignals: vis });
  },

  setActiveProduct: (p) => set({ activeProduct: p, activeTimestamp: null }),
  setActiveTimestamp: (t) => set({ activeTimestamp: t }),
  setPrefs: (p) =>
    set({
      prefs: p,
      axisAssignments: p.axisAssignments ?? {}
    }),

  toggleSignal: (name) => {
    const next = new Set(get().visibleSignals);
    if (next.has(name)) next.delete(name);
    else next.add(name);
    set({ visibleSignals: next });
  },

  setSignalVisible: (name, visible) => {
    const next = new Set(get().visibleSignals);
    if (visible) next.add(name);
    else next.delete(name);
    set({ visibleSignals: next });
  },

  setAxis: (name, axis) => {
    const next = { ...get().axisAssignments, [name]: axis };
    set({ axisAssignments: next });
    const prefs = get().prefs;
    if (prefs) {
      const merged = { ...prefs, axisAssignments: next };
      set({ prefs: merged });
      window.api.setPrefs({ axisAssignments: next });
    }
  },

  setSignalColor: (name, color) => {
    const prefs = get().prefs;
    if (!prefs) return;
    const signalColors = { ...(prefs.signalColors ?? {}), [name]: color };
    const merged = { ...prefs, signalColors };
    set({ prefs: merged });
    window.api.setPrefs({ signalColors });
  },

  toggleMetric: (key) => {
    const prefs = get().prefs;
    if (!prefs) return;
    const has = prefs.enabledMetrics.includes(key);
    const enabledMetrics = has
      ? prefs.enabledMetrics.filter((k) => k !== key)
      : [...prefs.enabledMetrics, key];
    const merged = { ...prefs, enabledMetrics };
    set({ prefs: merged });
    window.api.setPrefs({ enabledMetrics });
  },

  setTheme: (t) => {
    const prefs = get().prefs;
    if (!prefs) return;
    const merged = { ...prefs, theme: t };
    set({ prefs: merged });
    window.api.setPrefs({ theme: t });
  }
}));
