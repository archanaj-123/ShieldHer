"""
api_server.py — Flask REST API
GET  /status    → live risk, alerts, person count
GET  /incidents → saved clips
POST /command   → {"cmd": "reset_risk"|"start_rec"|"stop_rec"}
"""

import threading, time
from flask import Flask, jsonify, request

app   = Flask(__name__)
_lock = threading.Lock()
_st   = dict(risk=0,emergency=False,recording=False,
             alerts=[],persons=0,url=None,ts=None)
_cmds = []

@app.route("/status")
def status():
    with _lock: return jsonify(_st)

@app.route("/incidents")
def incidents():
    import os; from pathlib import Path
    files=sorted(Path("incidents").glob("*.avi"),
                 key=os.path.getmtime,reverse=True)
    return jsonify([str(f) for f in files[:20]])

@app.route("/command",methods=["POST"])
def command():
    cmd=(request.json or {}).get("cmd","")
    if cmd: _cmds.append(cmd); return jsonify({"ok":True})
    return jsonify({"ok":False}),400

def update(risk,emergency,recording,alerts,persons,url=None):
    with _lock:
        _st.update(risk=risk,emergency=emergency,recording=recording,
                   alerts=[{"text":a.text,"sev":a.severity} for a in alerts],
                   persons=persons,url=url,ts=time.strftime("%H:%M:%S"))

def pop_cmds():
    c=list(_cmds); _cmds.clear(); return c

def start(host="0.0.0.0",port=5000):
    threading.Thread(
        target=lambda:app.run(host=host,port=port,debug=False,use_reloader=False),
        daemon=True).start()
    print(f"[API] http://{host}:{port}")
