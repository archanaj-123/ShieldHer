"""
main.py — ShieldHer AI Drone Safety System
──────────────────────────────────────────
python main.py                    → webcam
python main.py --source video.mp4 → video file
python main.py --no-api           → no Flask server

Keys:
  Q → quit
  R → record / stop manually
  S → toggle debug panel

Logs are saved to:  logs/run_YYYYMMDD_HHMMSS.log
Terminal only shows startup messages + errors.
──────────────────────────────────────────
"""

import cv2, sys, time, argparse, logging
from datetime import datetime
from pathlib import Path

from detection_engine  import DetectionEngine, Config
from incident_recorder import IncidentRecorder
from hud_renderer      import HUDRenderer
from telegram_alert    import get_bot
import api_server


# ── Logging — FILE ONLY, terminal stays clean ─────────────────
Path("logs").mkdir(exist_ok=True)
log_file = Path("logs") / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# File handler — gets everything (DEBUG+)
file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
))

# Terminal handler — only CRITICAL errors (startup failures, crashes)
term_handler = logging.StreamHandler(sys.stdout)
term_handler.setLevel(logging.CRITICAL)
term_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, term_handler])
log = logging.getLogger("ShieldHer")

# Startup prints go to terminal directly (not through logging)
def info(msg):
    print(msg)
    log.info(msg)


# ── Alert deduplication ───────────────────────────────────────
_last_log: dict = {}

def _log_event(key, msg, level="info", cooldown=6.0):
    now = time.time()
    if now - _last_log.get(key, 0) >= cooldown:
        _last_log[key] = now
        getattr(log, level)(msg)


# ── Args ──────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--source", default="0")
ap.add_argument("--no-api", action="store_true")
args   = ap.parse_args()
source = int(args.source) if args.source.isdigit() else args.source


# ── Init ──────────────────────────────────────────────────────
info("=" * 55)
info("  ShieldHer AI — Drone Safety System")
info("=" * 55)
info(f"Log file → {log_file}")

engine   = DetectionEngine()
recorder = IncidentRecorder(fps=20, size=(Config.W, Config.H))
hud      = HUDRenderer()
bot      = get_bot()

if not args.no_api:
    api_server.start()

cap = cv2.VideoCapture(source)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.H)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

if not cap.isOpened():
    print(f"ERROR: Cannot open camera source: {source}")
    sys.exit(1)

info(f"Camera OK  source={source}  {Config.W}x{Config.H}")
info("Keys: Q=quit  R=record  S=debug panel")
info("All detection logs → " + str(log_file))
info("-" * 55)


# ── State ─────────────────────────────────────────────────────
COOLDOWN     = 12
last_emerg_t = 0
frame_n      = 0
show_debug   = False
fps_t        = time.time()
fps_n        = 0


# ── Main loop ─────────────────────────────────────────────────
while True:
    cap.grab()
    ret, frame = cap.retrieve()
    if not ret:
        ret, frame = cap.read()
    if not ret:
        print("Camera stream ended.")
        break

    frame   = cv2.resize(frame, (Config.W, Config.H))
    frame_n += 1
    fps_n   += 1

    annotated, persons, g_alerts, risk = engine.process(
        frame, run_obj=(frame_n % 3 == 0)
    )

    now       = time.time()
    emergency = risk >= Config.EMERGENCY_THRESH

    # ── Log FPS status every 5s (file only) ───────────────────
    if now - fps_t >= 5.0:
        fps = fps_n / (now - fps_t)
        fps_n, fps_t = 0, now
        log.debug(f"STATUS fps={fps:.1f} persons={len(persons)}"
                  f" risk={risk} rec={recorder.is_recording}")

    # ── Log person alerts (file only) ─────────────────────────
    for s in persons:
        for a in s.alerts:
            if a.severity == "info":
                continue   # skip info-level (e.g. "Hands Raised 0.5s")
            key = f"P{s.pid}:{a.text}"
            _log_event(
                key,
                f"[P{s.pid}] {a.text}"
                f" | tilt={s.tilt:.0f}°"
                f" wv={s.wrist_vel:.0f}"
                f" tv={s.torso_vel:.0f}"
                f" lying={s.is_lying}"
                f" lstm={s.lstm_label}({s.lstm_conf:.0%})"
                f" buf={s.buf_size}/30",
                level="warning" if a.severity == "critical" else "info"
            )

    # ── Log global alerts (file only) ─────────────────────────
    for a in g_alerts:
        lvl = "warning" if a.severity == "critical" else "info"
        _log_event(a.text, f"[GLOBAL] {a.text} +{a.risk}risk", lvl)

    # ── Log emergency (file only) ─────────────────────────────
    if emergency:
        _log_event(
            "__EMERG__",
            f"!!! EMERGENCY risk={risk} persons={len(persons)} !!!",
            "warning", cooldown=10.0
        )

    # ── Telegram alerts ───────────────────────────────────────
    # Send per-person critical alerts with snapshot
    for s in persons:
        for a in s.alerts:
            if a.severity in ("critical", "warning") and a.risk > 0:
                bot.notify(
                    alert_text = a.text,
                    severity   = a.severity,
                    risk       = risk,
                    frame      = annotated.copy() if a.severity == "critical" else None,
                    extra      = f"Person {s.pid} | tilt={s.tilt:.0f}° lying={s.is_lying}"
                )

    # Send global alerts (fight, harassment, weapon etc.)
    for a in g_alerts:
        if a.severity in ("critical", "warning") and a.risk > 0:
            bot.notify(
                alert_text = a.text,
                severity   = a.severity,
                risk       = risk,
                frame      = annotated.copy() if a.severity == "critical" else None,
            )

    # Full emergency escalation message
    if emergency:
        bot.notify_emergency(risk, g_alerts, frame=annotated.copy())

    # ── Recording ─────────────────────────────────────────────
    if emergency:
        last_emerg_t = now
        if not recorder.is_recording:
            label = g_alerts[0].text if g_alerts else "EMERGENCY"
            recorder.start(label)
            log.warning(f"[REC] Started trigger={label}")
    else:
        if recorder.is_recording and now - last_emerg_t > COOLDOWN:
            recorder.stop()
            log.info("[REC] Stopped — risk cleared.")

    recorder.write(annotated)

    # Notify telegram when clip is uploaded to Drive
    if recorder.last_url:
        _log_event("__URL__", f"[Drive] {recorder.last_url}", "info")
        bot.notify_clip(recorder.last_url,
                        g_alerts[0].text if g_alerts else "incident")

    # ── Draw ──────────────────────────────────────────────────
    final = hud.draw(
        annotated, persons, g_alerts, risk,
        recorder.is_recording,
        upload_url = recorder.last_url,
        show_debug = show_debug,
        lstm       = engine.lstm,
    )

    # ── API ───────────────────────────────────────────────────
    if not args.no_api:
        api_server.update(risk, emergency, recorder.is_recording,
                          g_alerts, len(persons), recorder.last_url)
        for cmd in api_server.pop_cmds():
            if cmd == "reset_risk":
                engine.mem.risk_hist.clear()
                log.info("[CMD] Risk reset.")
            elif cmd == "start_rec":
                recorder.start("manual")
            elif cmd == "stop_rec":
                recorder.stop()

    # ── Display ───────────────────────────────────────────────
    cv2.imshow("ShieldHer AI", final)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        log.info("Quit."); break
    elif key == ord("r"):
        if recorder.is_recording:
            recorder.stop(); log.info("[REC] Manual stop.")
        else:
            recorder.start("manual"); log.info("[REC] Manual start.")
    elif key == ord("s"):
        show_debug = not show_debug
        log.info(f"Debug {'ON' if show_debug else 'OFF'}")


# ── Cleanup ───────────────────────────────────────────────────
if recorder.is_recording:
    recorder.stop()
cap.release()
cv2.destroyAllWindows()
info(f"Shutdown. Full log → {log_file}")
