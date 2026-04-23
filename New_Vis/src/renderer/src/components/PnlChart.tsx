import { useEffect, useRef, useState } from 'react';
import * as echarts from 'echarts';
import type { EChartsOption } from 'echarts';
import type { ProductData } from '../../../preload/index.js';

interface Props {
  data: ProductData;
  isDark: boolean;
}

export function PnlChart({ data, isDark }: Props) {
  const rootRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);
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
      const pnl = data.columns['profit_and_loss'] ?? [];
      const pos = data.columns['position'] ?? [];
      const drawdown: number[] = [];
      let peak = pnl[0] ?? 0;
      for (let i = 0; i < ts.length; i++) {
        const v = pnl[i] ?? 0;
        if (v > peak) peak = v;
        drawdown.push(v - peak);
      }

      const option: EChartsOption = {
        animation: false,
        grid: { left: 56, right: 74, top: 10, bottom: 48, containLabel: true },
        tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
        xAxis: { type: 'value', scale: true, axisLabel: { color: '#94a3b8' } },
        yAxis: [
          { type: 'value', name: 'PnL', scale: true, axisLabel: { color: '#94a3b8' } },
          { type: 'value', name: 'Position', scale: true, axisLabel: { color: '#94a3b8' } },
        ],
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
            yAxisIndex: [0, 1],
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
            yAxisIndex: [0, 1],
            width: 12,
            right: 8,
            filterMode: 'none',
            borderColor: '#334155',
            backgroundColor: 'rgba(148,163,184,0.08)',
          },
        ],
        series: [
          {
            name: 'profit_and_loss',
            type: 'line',
            showSymbol: false,
            lineStyle: { color: '#10b981', width: 1.5 },
            data: ts.map((t: number, i: number) => [t, pnl[i] ?? 0]),
          },
          {
            name: 'drawdown',
            type: 'line',
            showSymbol: false,
            lineStyle: { color: '#f43f5e', width: 1 },
            areaStyle: { color: 'rgba(244,63,94,0.18)' },
            data: ts.map((t: number, i: number) => [t, drawdown[i]]),
          },
          {
            name: 'position',
            type: 'line',
            yAxisIndex: 1,
            showSymbol: false,
            lineStyle: { color: '#6366f1', width: 1.2 },
            data: ts.map((t: number, i: number) => [t, pos[i] ?? 0]),
          },
        ],
      };
      chart.setOption(option, { notMerge: true, lazyUpdate: true });

      resizeObserver = new ResizeObserver(() => chart.resize());
      resizeObserver.observe(rootEl);
      setIsReady(true);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setChartError(`PnL chart failed to initialize: ${message}`);
    }

    return () => {
      resizeObserver?.disconnect();
      chartRef.current?.dispose();
      chartRef.current = null;
    };
  }, [data, isDark]);

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <div ref={rootRef} style={{ width: '100%', height: '100%' }} />
      {!isReady && !chartError && (
        <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', pointerEvents: 'none' }}>
          Loading PnL chart...
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
