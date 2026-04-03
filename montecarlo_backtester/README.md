This was cloned from https://github.com/chrispyroberts/imc-prosperity-4

Following changes were made 

visualizer\vite.config.ts
```
...
export default defineConfig({
  plugins: [react()],
  base: '/',
  server: {
    port: 5555, #added this port statement
    proxy: {
...

``` 
backtester\pyproject.toml

removed the following to be able to install as a uv packaged
```
[tool.uv.workspace]
members = [
    ".",
]

[tool.uv.sources]
prosperity4mcbt = { workspace = true, editable = true }

```
backtester\prosperity4mcbt\open.py
added this to auto run frontend
```
    cwd = Path(__file__).resolve().parents[2] / "visualizer"
    system = platform.system()

    if system == "Darwin":  # macOS
        subprocess.Popen([
            "osascript", "-e",
            f'tell app "Terminal" to do script "cd {cwd} && npm run dev"'
        ])
    elif system == "Windows":
        subprocess.Popen(
            ["cmd", "/c", "start", "cmd", "/k", "npm run dev"],
            cwd=cwd
        )
    else:  # Linux
        subprocess.Popen(
            ["xterm", "-e", "npm run dev"],
            cwd=cwd
        )
```
