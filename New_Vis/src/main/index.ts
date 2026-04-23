import { app, BrowserWindow, dialog, ipcMain, shell } from 'electron';
import path from 'node:path';
import { readdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { parseLog } from './parser.js';
import type { Prefs } from '../preload/types.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const DEFAULT_PREFS: Prefs = {
  theme: 'dark',
  enabledMetrics: [
    'final_pnl',
    'sharpe',
    'max_drawdown',
    'volatility',
    'position_skew',
    'fill_rate',
    'total_volume'
  ],
  axisAssignments: {},
  signalColors: {},
  lastLogDir: undefined
};

let prefsStore: { get: () => Prefs; set: (p: Prefs) => void };

async function initPrefs(): Promise<void> {
  const mod = await import('electron-store');
  const Store = (mod as any).default ?? mod;
  const store = new Store({ name: 'prosperity-vis-prefs', defaults: DEFAULT_PREFS });
  prefsStore = {
    get: () => ({ ...DEFAULT_PREFS, ...(store as any).store }) as Prefs,
    set: (p: Prefs) => {
      (store as any).store = p;
    }
  };
}

function createWindow(): BrowserWindow {
  const win = new BrowserWindow({
    width: 1600,
    height: 1000,
    show: false,
    backgroundColor: '#0f172a',
    webPreferences: {
      preload: path.join(__dirname, '../preload/index.mjs'),
      sandbox: false,
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  win.on('ready-to-show', () => win.show());

  win.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });

  if (process.env.ELECTRON_RENDERER_URL) {
    win.loadURL(process.env.ELECTRON_RENDERER_URL);
  } else {
    win.loadFile(path.join(__dirname, '../renderer/index.html'));
  }
  return win;
}

function registerIpc(): void {
  ipcMain.handle('log:open-dialog', async () => {
    const prefs = prefsStore.get();
    const result = await dialog.showOpenDialog({
      title: 'Select a Prosperity log file',
      filters: [{ name: 'Log files', extensions: ['log', 'json'] }, { name: 'All files', extensions: ['*'] }],
      properties: ['openFile'],
      defaultPath: prefs.lastLogDir
    });
    if (result.canceled || result.filePaths.length === 0) return null;
    const p = result.filePaths[0];
    prefsStore.set({ ...prefs, lastLogDir: path.dirname(p) });
    return parseLog(p);
  });

  ipcMain.handle('log:load-path', async (_e, p: string) => parseLog(p));

  ipcMain.handle('log:list-dir', async (_e, dir: string) => {
    try {
      const entries = await readdir(dir, { withFileTypes: true });
      return entries
        .filter((e) => e.isFile() && (e.name.endsWith('.log') || e.name.endsWith('.json')))
        .map((e) => ({ path: path.join(dir, e.name), name: e.name }))
        .sort((a, b) => a.name.localeCompare(b.name));
    } catch {
      return [];
    }
  });

  ipcMain.handle('prefs:get', async () => prefsStore.get());
  ipcMain.handle('prefs:set', async (_e, patch: Partial<Prefs>) => {
    const merged = { ...prefsStore.get(), ...patch };
    prefsStore.set(merged);
    return merged;
  });
}

app.whenReady().then(async () => {
  await initPrefs();
  registerIpc();
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
