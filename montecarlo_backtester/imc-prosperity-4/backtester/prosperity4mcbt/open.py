import webbrowser
from pathlib import Path
import subprocess
import platform
from prosperity4mcbt.dashboard_server import ensure_dashboard_server


def open_dashboard(output_file: Path) -> None:
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
        
    ensure_dashboard_server(output_file.parent)
    webbrowser.open("http://localhost:5555/")
    
