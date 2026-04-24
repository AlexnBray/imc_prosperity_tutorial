import { contextBridge, ipcRenderer } from 'electron';
import type { Api, Prefs } from './types.js';

const api: Api = {
  openLogDialog: () => ipcRenderer.invoke('log:open-dialog'),
  loadLogByPath: (p: string) => ipcRenderer.invoke('log:load-path', p),
  listLogsInDir: (d: string) => ipcRenderer.invoke('log:list-dir', d),
  getPrefs: () => ipcRenderer.invoke('prefs:get'),
  setPrefs: (patch: Partial<Prefs>) => ipcRenderer.invoke('prefs:set', patch)
};

contextBridge.exposeInMainWorld('api', api);
