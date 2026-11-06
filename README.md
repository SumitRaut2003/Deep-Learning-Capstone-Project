# Helmet Compliance Monitor (web UI)

1. Copy `best.onnx` into this folder (next to `app.py`).
2. `pip install -r requirements.txt`
3. `python app.py`  (or double-click `run_web.bat`)
4. Open http://127.0.0.1:5000 and click **Start camera**.

Violations are saved to `snapshots/` and logged in `violations.csv`.
The camera only works on `localhost`/`127.0.0.1` (or HTTPS). Stop the server with Ctrl+C.
