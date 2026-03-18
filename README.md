# Getting Started

**Quick Setup**

1. Install [uv](https://docs.astral.sh/uv/) if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone and sync dependencies:
   ```bash
   git clone <repository>
   cd <project>
   uv sync
   ```

3. Activate the virtual environment:
   ```bash
   # macOS/Linux
   source .venv/bin/activate
   
   # Windows
   .venv\Scripts\activate
   ```

That's it—`uv sync` creates the venv automatically and installs all dependencies from `pyproject.toml` and `uv.lock`.


For Alex:
C:\Users\alexn\AppData\Local\Python\pythoncore-3.14-64\Scripts\uv.exe sync
