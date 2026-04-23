import { readFile } from 'node:fs/promises';
import path from 'node:path';
import type { ParsedLog, ProductData, Trade } from '../preload/types.js';

interface ProsperityLog {
  activitiesLog?: string;
  logs?: Array<{ lambdaLog?: string; sandboxLog?: string }>;
  tradeHistory?: Array<{
    timestamp: number;
    symbol: string;
    price: number;
    quantity: number;
    buyer: string;
    seller: string;
  }>;
}

const BASE_COLUMNS = new Set([
  'day',
  'timestamp',
  'product',
  'bid_price_1', 'bid_price_2', 'bid_price_3',
  'bid_volume_1', 'bid_volume_2', 'bid_volume_3',
  'ask_price_1', 'ask_price_2', 'ask_price_3',
  'ask_volume_1', 'ask_volume_2', 'ask_volume_3',
  'mid_price',
  'profit_and_loss',
  'position',
  'algo_bid_price_1', 'algo_bid_price_2', 'algo_bid_price_3',
  'algo_bid_volume_1', 'algo_bid_volume_2', 'algo_bid_volume_3',
  'algo_ask_price_1', 'algo_ask_price_2', 'algo_ask_price_3',
  'algo_ask_volume_1', 'algo_ask_volume_2', 'algo_ask_volume_3',
  'algo_bid_fill_1_price', 'algo_bid_fill_2_price', 'algo_bid_fill_3_price',
  'algo_bid_fill_1_volume', 'algo_bid_fill_2_volume', 'algo_bid_fill_3_volume',
  'algo_ask_fill_1_price', 'algo_ask_fill_2_price', 'algo_ask_fill_3_price',
  'algo_ask_fill_1_volume', 'algo_ask_fill_2_volume', 'algo_ask_fill_3_volume',
  'market_bid_fill_1_price', 'market_bid_fill_2_price', 'market_bid_fill_3_price',
  'market_bid_fill_1_volume', 'market_bid_fill_2_volume', 'market_bid_fill_3_volume',
  'market_ask_fill_1_price', 'market_ask_fill_2_price', 'market_ask_fill_3_price',
  'market_ask_fill_1_volume', 'market_ask_fill_2_volume', 'market_ask_fill_3_volume',
]);

function parseCsv(csv: string): Array<Record<string, string>> {
  const rows: Array<Record<string, string>> = [];
  const lines = csv.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length === 0) return rows;
  const header = lines[0].split(';');
  for (let i = 1; i < lines.length; i++) {
    const cells = lines[i].split(';');
    const row: Record<string, string> = {};
    for (let j = 0; j < header.length; j++) {
      row[header[j]] = cells[j] ?? '';
    }
    rows.push(row);
  }
  return rows;
}

function num(v: string | undefined): number {
  if (v === undefined || v === '' || v === 'nan' || v === 'NaN') return 0;
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
}

interface AlgoRecord {
  timestamp: number;
  product: string;
  position: number;
  extras: Record<string, number>; // fv, effective_fv, secondary_fv, and any future signals
  bids: Array<{ price: number; volume: number }>;
  asks: Array<{ price: number; volume: number }>;
}

// ALGO line format from the Python parser:
//   [ALGO],<timestamp>,<product>,<position>,<fv>,<effective_fv>,<secondary_fv>,[p:v;p:v;p:v],[p:v;p:v;p:v]
// We keep the three numeric signal columns but name them so they show up as "signalColumns"
// and allow a future extension with JSON-keyed metadata.
const SIGNAL_NAMES = ['fv', 'effective_fv', 'secondary_fv'];

function parseAlgoLines(lambdaLogs: string[]): AlgoRecord[] {
  const out: AlgoRecord[] = [];
  for (const block of lambdaLogs) {
    for (const line of block.split('\n')) {
      if (!line.startsWith('[ALGO]')) continue;
      const parts = line.split(',');
      if (parts.length < 7) continue;
      const rec: AlgoRecord = {
        timestamp: parseInt(parts[1], 10),
        product: parts[2],
        position: parseInt(parts[3], 10),
        extras: {},
        bids: [],
        asks: []
      };
      for (let i = 0; i < SIGNAL_NAMES.length; i++) {
        const raw = parts[4 + i];
        if (raw !== undefined) rec.extras[SIGNAL_NAMES[i]] = num(raw);
      }
      const parseBook = (s: string | undefined) => {
        if (!s) return [];
        const inner = s.trim().replace(/^\[|\]$/g, '');
        if (!inner) return [];
        return inner.split(';').map((pv) => {
          const [p, v] = pv.split(':');
          return { price: num(p), volume: Math.abs(num(v)) };
        });
      };
      rec.bids = parseBook(parts[7]);
      rec.asks = parseBook(parts[8]);
      out.push(rec);
    }
  }
  return out;
}

export async function parseLog(filePath: string): Promise<ParsedLog> {
  const raw = await readFile(filePath, 'utf-8');
  const data = JSON.parse(raw) as ProsperityLog;

  const marketRows = parseCsv(data.activitiesLog ?? '');
  const algoRecords = parseAlgoLines(
    (data.logs ?? []).map((l) => l.lambdaLog ?? '')
  );
  const trades = data.tradeHistory ?? [];

  // Index algo records by (timestamp, product)
  const algoIx = new Map<string, AlgoRecord>();
  for (const r of algoRecords) {
    algoIx.set(`${r.timestamp}|${r.product}`, r);
  }

  // Build per-product columnar store keyed from market rows
  const perProduct: Record<string, { ts: number[]; cols: Record<string, number[]> }> = {};
  const allExtraKeys = new Set<string>();

  const ensureCol = (
    store: { ts: number[]; cols: Record<string, number[]> },
    key: string,
    length: number,
    fill = 0
  ) => {
    if (!store.cols[key]) {
      store.cols[key] = new Array(length).fill(fill);
    }
  };

  for (const row of marketRows) {
    const product = row['product'];
    if (!product) continue;
    const ts = parseInt(row['timestamp'], 10);
    if (!Number.isFinite(ts)) continue;

    if (!perProduct[product]) perProduct[product] = { ts: [], cols: {} };
    const store = perProduct[product];
    const rowIdx = store.ts.length;
    store.ts.push(ts);

    // Copy every numeric column we see from the activitiesLog CSV
    for (const [k, v] of Object.entries(row)) {
      if (k === 'product' || k === 'timestamp') continue;
      const n = num(v);
      if (!store.cols[k]) store.cols[k] = new Array(rowIdx).fill(0);
      store.cols[k].push(n);
    }
    // Any column declared earlier but missing in this row: pad 0
    for (const key of Object.keys(store.cols)) {
      if (store.cols[key].length === rowIdx) store.cols[key].push(0);
    }

    // Merge algo record
    const algo = algoIx.get(`${ts}|${product}`);
    if (algo) {
      const writeCell = (key: string, val: number) => {
        ensureCol(store, key, rowIdx);
        // Pad any trailing gap then assign at rowIdx
        while (store.cols[key].length < rowIdx) store.cols[key].push(0);
        store.cols[key][rowIdx] = val;
      };
      writeCell('position', algo.position);
      for (const [k, v] of Object.entries(algo.extras)) {
        writeCell(k, v);
        if (!BASE_COLUMNS.has(k)) allExtraKeys.add(k);
      }
      for (let i = 0; i < 3; i++) {
        if (algo.bids[i]) {
          writeCell(`algo_bid_price_${i + 1}`, algo.bids[i].price);
          writeCell(`algo_bid_volume_${i + 1}`, algo.bids[i].volume);
        }
        if (algo.asks[i]) {
          writeCell(`algo_ask_price_${i + 1}`, algo.asks[i].price);
          writeCell(`algo_ask_volume_${i + 1}`, algo.asks[i].volume);
        }
      }
    }
  }

  // Post-pass: square up all column lengths per product
  for (const store of Object.values(perProduct)) {
    const len = store.ts.length;
    for (const key of Object.keys(store.cols)) {
      while (store.cols[key].length < len) store.cols[key].push(0);
    }
  }

  // Vectorised trade-to-fill attribution (mirrors log_parser.py)
  const parsedTrades: Trade[] = [];
  for (const t of trades) {
    const product = t.symbol;
    const isAlgo = t.buyer === 'SUBMISSION' || t.seller === 'SUBMISSION';
    const algoSide: 'buy' | 'sell' | null = isAlgo
      ? t.buyer === 'SUBMISSION'
        ? 'buy'
        : 'sell'
      : null;
    parsedTrades.push({
      timestamp: t.timestamp,
      product,
      price: t.price,
      quantity: t.quantity,
      buyer: t.buyer,
      seller: t.seller,
      algoSide
    });

    // Project trade onto fill columns for the chart overlay
    const store = perProduct[product];
    if (!store) continue;
    // Binary search would be better; market rows are sorted, linear-last OK for MVP
    // We rely on timestamps being unique per product per row
    const idx = store.ts.indexOf(t.timestamp);
    if (idx < 0) continue;

    const role = isAlgo ? 'algo' : 'market';
    // side_type: which side of the book the fill occurred on
    // SUBMISSION buyer => we bought at the ask
    let side: 'bid' | 'ask' = t.buyer === 'SUBMISSION' ? 'ask' : 'bid';
    let level = 3;
    for (let lvl = 1; lvl <= 3; lvl++) {
      const bidKey = role === 'algo' ? `algo_bid_price_${lvl}` : `bid_price_${lvl}`;
      const askKey = role === 'algo' ? `algo_ask_price_${lvl}` : `ask_price_${lvl}`;
      const bidPx = store.cols[bidKey]?.[idx] ?? 0;
      const askPx = store.cols[askKey]?.[idx] ?? 0;
      if (t.price === bidPx) { side = 'bid'; level = lvl; break; }
      if (t.price === askPx) { side = 'ask'; level = lvl; break; }
    }
    const vCol = `${role}_${side}_fill_${level}_volume`;
    const pCol = `${role}_${side}_fill_${level}_price`;
    if (!store.cols[vCol]) store.cols[vCol] = new Array(store.ts.length).fill(0);
    if (!store.cols[pCol]) store.cols[pCol] = new Array(store.ts.length).fill(0);
    store.cols[vCol][idx] = (store.cols[vCol][idx] ?? 0) + t.quantity;
    store.cols[pCol][idx] = t.price;
  }

  // Finalise output structures + compute signal vs base classification
  const finalPerProduct: Record<string, ProductData> = {};
  const signalSet = new Set<string>();
  const baseSet = new Set<string>();
  for (const [product, store] of Object.entries(perProduct)) {
    finalPerProduct[product] = {
      product,
      timestamps: store.ts,
      columns: store.cols
    };
    for (const k of Object.keys(store.cols)) {
      if (BASE_COLUMNS.has(k)) baseSet.add(k);
      else signalSet.add(k);
    }
  }

  return {
    path: filePath,
    name: path.basename(filePath),
    products: Object.keys(finalPerProduct).sort(),
    perProduct: finalPerProduct,
    trades: parsedTrades,
    baseColumns: [...baseSet].sort(),
    signalColumns: [...signalSet].sort()
  };
}
