export interface ProductData {
  product: string;
  timestamps: number[];
  columns: Record<string, number[]>;
}

export interface Trade {
  timestamp: number;
  product: string;
  price: number;
  quantity: number;
  buyer: string;
  seller: string;
  algoSide: 'buy' | 'sell' | null;
}

export interface ParsedLog {
  path: string;
  name: string;
  products: string[];
  perProduct: Record<string, ProductData>;
  trades: Trade[];
  baseColumns: string[];
  signalColumns: string[];
}

export interface Prefs {
  theme: 'dark' | 'light';
  enabledMetrics: string[];
  axisAssignments: Record<string, 'y' | 'y2' | 'y3'>;
  signalColors: Record<string, string>;
  lastLogDir?: string;
}

export interface Api {
  openLogDialog: () => Promise<ParsedLog | null>;
  loadLogByPath: (path: string) => Promise<ParsedLog | null>;
  listLogsInDir: (dir: string) => Promise<{ path: string; name: string }[]>;
  getPrefs: () => Promise<Prefs>;
  setPrefs: (patch: Partial<Prefs>) => Promise<Prefs>;
}
