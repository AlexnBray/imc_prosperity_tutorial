import { useEffect, useRef, useState } from 'react';
import * as echarts from 'echarts';
import type { EChartsOption } from 'echarts';
import type { ProductData, Trade } from '../../../preload/index.js';
import { useApp } from '../store.js';

interface Props {
  data: ProductData;
  trades: Trade[];
  isDark: boolean;
}

const DEFAULT_SIGNAL_COLORS: Record<string, string> = {
  mid_price: '#64748b',
  fv: '#ffffff',
  effective_fv: '#ff00d4',
  secondary_fv: '#00ccff',
};

function getSignalColor(name: string, userColors: Record<string, string>): string {
  return userColors[name] ?? DEFAULT_SIGNAL_COLORS[name] ?? '#94a3b8';
}

export function PriceChart({ data, trades: _trades, isDark }: Props) {
  const rootRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);
  const { prefs, visibleSignals, setActiveTimestamp } = useApp();
  const [chartError, setChartError] = useState<string | null>(null);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    if (!rootRef.current) return;
    let resizeObserver: ResizeObserver | null = null;
    setChartError(null);
    setIsReady(false);

    try {
      const rootEl = rootRef.current;
      const chart = echarts.init(rootEl, isDark ? 'dark' : undefined);
      chart.group = 'main-charts';
      echarts.connect('main-charts');
      chartRef.current = chart;

      const ts = data.timestamps;
      const cols = data.columns;
      const userColors = prefs?.signalColors ?? {};

      const orderedKeys = [
        'mid_price',
        'bid_price_1', 'ask_price_1',
        'bid_price_2', 'ask_price_2',
        'bid_price_3', 'ask_price_3',
        'algo_bid_price_1', 'algo_ask_price_1',
        'algo_bid_price_2', 'algo_ask_price_2',
        'algo_bid_price_3', 'algo_ask_price_3',
        'fv', 'effective_fv', 'secondary_fv',
      ];

      const candidateKeys = [
        ...orderedKeys,
        ...Object.keys(cols).filter((k) => !orderedKeys.includes(k)),
      ];

      const series = [...new Set(candidateKeys)]
        .filter((k) => cols[k] && k !== 'profit_and_loss' && k !== 'position' && k !== 'day')
        .filter((key) => {
          const isDefaultVisible = key === 'mid_price' || key === 'bid_price_1' || key === 'ask_price_1';
          return visibleSignals.has(key) || isDefaultVisible;
        })
        .map((key) => {
          const values = cols[key];
          return {
            name: key,
            type: 'line' as const,
            showSymbol: false,
            connectNulls: true,
            smooth: false,
            lineStyle: { width: key === 'mid_price' ? 2 : 1.2, color: getSignalColor(key, userColors) },
            data: ts.map((t: number, i: number) => [t, values[i] ?? null]),
          };
        });

      if (series.length === 0) {
        setChartError('Price chart has no plottable columns for this product.');
        return;
      }

      const option: EChartsOption = {
        animation: false,
        grid: { left: 56, right: 54, top: 12, bottom: 52, containLabel: true },
        tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
        legend: { show: false },
        xAxis: { type: 'value', scale: true, axisLabel: { color: '#94a3b8' } },
        yAxis: { type: 'value', scale: true, axisLabel: { color: '#94a3b8' } },
        axisPointer: { link: [{ xAxisIndex: 'all' }] },
        toolbox: {
          show: true,
          top: 6,
          right: 8,
          itemSize: 12,
          feature: { restore: {}, saveAsImage: {} },
        },
        dataZoom: [
          {
            type: 'inside',
            xAxisIndex: 0,
            filterMode: 'none',
            zoomOnMouseWheel: true,
            moveOnMouseMove: true,
            moveOnMouseWheel: true,
          },
          {
            type: 'inside',
            yAxisIndex: 0,
            filterMode: 'none',
            zoomOnMouseWheel: 'shift',
            moveOnMouseMove: 'shift',
            moveOnMouseWheel: false,
          },
          {
            type: 'slider',
            xAxisIndex: 0,
            height: 16,
            bottom: 8,
            filterMode: 'none',
            borderColor: '#334155',
            backgroundColor: 'rgba(148,163,184,0.08)',
          },
          {
            type: 'slider',
            yAxisIndex: 0,
            width: 12,
            right: 8,
            filterMode: 'none',
            borderColor: '#334155',
            backgroundColor: 'rgba(148,163,184,0.08)',
          },
        ],
        series,
      };
      chart.setOption(option, { notMerge: true, lazyUpdate: true });

      chart.on('updateAxisPointer', (params: unknown) => {
        const p = params as { axesInfo?: Array<{ value?: number }> };
        const xVal = p?.axesInfo?.[0]?.value;
        if (xVal !== undefined) setActiveTimestamp(xVal);
      });
      chart.on('mouseout', () => setActiveTimestamp(null));

      resizeObserver = new ResizeObserver(() => chart.resize());
      resizeObserver.observe(rootEl);
      setIsReady(true);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setChartError(`Price chart failed to initialize: ${message}`);
    }

    return () => {
      resizeObserver?.disconnect();
      chartRef.current?.dispose();
      chartRef.current = null;
    };
  }, [data, isDark, prefs?.signalColors, visibleSignals]);

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <div ref={rootRef} style={{ width: '100%', height: '100%' }} />
      {!isReady && !chartError && (
        <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', pointerEvents: 'none' }}>
          Loading price chart...
        </div>
      )}
      {chartError && (
        <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#f43f5e', padding: 12, textAlign: 'center' }}>
          {chartError}
        </div>
      )}
    </div>
  );
}
