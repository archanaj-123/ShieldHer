"""
telegram_alert.py
Sends concise police-format crime alerts to Telegram.
Two channels: personal (quick emoji) + police (formal).
"""

import requests, threading, time, cv2, tempfile, os
from datetime import datetime

# ── Credentials ───────────────────────────────────────────────
TOKEN          = "8609692022:AAG5mNPoF1jrNGyZkGU9uCs5AxFkXQ9occQ"
CHAT_ID        = "6044884740"   # your personal chat
POLICE_CHAT_ID = "6044884740"   # replace with police group chat ID

# ── Camera info — fill these in ───────────────────────────────
LOCATION  = "Location not set"   # e.g. "Anna Nagar, Chennai - Gate B"
CAMERA_ID = "CAM-01"
CONTACT   = "ShieldHer AI System"

BASE      = f"https://api.telegram.org/bot{TOKEN}"

# Cooldown per alert type (seconds)
COOLDOWN  = {
    "EMERGENCY": 20, "ASSAULT": 15, "FIGHTING": 15,
    "WEAPON": 10,    "KNIFE": 10,   "PERSON DOWN": 15,
    "FAINTED": 15,   "HARASSMENT": 25, "GRABBING": 20,
    "FOLLOWING": 30, "CORNERING": 20,  "SOS": 20,
    "DEFAULT": 30,
}

# Incident ID counter
_incident_n = 0
def _new_id():
    global _incident_n
    _incident_n += 1
    return f"SH-{datetime.now().strftime('%Y%m%d')}-{_incident_n:04d}"


class TelegramAlert:

    def __init__(self):
        self._last:  dict = {}
        self._queue: list = []
        self._lock        = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()
        print(f"[Telegram] Ready  chat={CHAT_ID}")

    # ── Called from main loop ─────────────────────────────────
    def notify(self, alert_text, severity, risk, frame=None, extra=""):
        key = next((k for k in COOLDOWN if k in alert_text.upper()), "DEFAULT")
        now = time.time()
        if now - self._last.get(key, 0) < COOLDOWN[key]:
            return
        self._last[key] = now

        iid = _new_id()
        ts  = datetime.now().strftime("%d %b %Y  %H:%M:%S")

        # ── Personal message (quick, emoji) ───────────────────
        sev_icon = "🔴" if severity == "critical" else "🟡"
        personal = (
            f"{sev_icon} *{alert_text}*\n"
            f"`{ts}` | Risk: `{risk}` | {CAMERA_ID}\n"
            f"ID: `{iid}`"
        )
        if extra:
            personal += f"\n_{extra}_"

        # ── Police message (formal, concise) ──────────────────
        police = (
            f"🚨 *CRIME ALERT — ShieldHer AI*\n"
            f"ID: `{iid}`\n"
            f"Time: `{ts}`\n"
            f"Location: {LOCATION}\n"
            f"Camera: {CAMERA_ID}\n"
            f"Incident: *{alert_text}*\n"
            f"Severity: {'HIGH' if severity == 'critical' else 'MEDIUM'}"
            f" | Risk: `{risk}/100`\n"
            f"Action: Immediate response required.\n"
            f"Contact: {CONTACT}"
        )

        with self._lock:
            # Personal — with snapshot if critical
            self._queue.append({
                "msg":     personal,
                "chat":    CHAT_ID,
                "frame":   frame if severity == "critical" else None,
            })
            # Police — always with snapshot
            self._queue.append({
                "msg":   police,
                "chat":  POLICE_CHAT_ID,
                "frame": frame,
            })

    def notify_emergency(self, risk, alerts, frame=None):
        key = "EMERGENCY"
        now = time.time()
        if now - self._last.get(key, 0) < COOLDOWN[key]:
            return
        self._last[key] = now

        iid  = _new_id()
        ts   = datetime.now().strftime("%d %b %Y  %H:%M:%S")
        acts = "  ".join(f"• {a.text}" for a in alerts[:3])

        police = (
            f"🚨🚨 *EMERGENCY — ShieldHer AI* 🚨🚨\n"
            f"ID: `{iid}`\n"
            f"Time: `{ts}`\n"
            f"Location: {LOCATION}\n"
            f"Camera: {CAMERA_ID}\n"
            f"Risk Score: `{risk}/100`\n"
            f"Detections: {acts}\n"
            f"*IMMEDIATE RESPONSE REQUIRED*\n"
            f"Contact: {CONTACT}"
        )
        with self._lock:
            self._queue.append({"msg": police, "chat": CHAT_ID,        "frame": frame})
            self._queue.append({"msg": police, "chat": POLICE_CHAT_ID, "frame": frame})

    def notify_clip(self, url, incident_type):
        key = f"__CLIP__{url}"
        if key in self._last:
            return
        self._last[key] = time.time()
        ts  = datetime.now().strftime("%H:%M:%S")
        msg = (
            f"📹 *Evidence Clip Ready*\n"
            f"`{ts}` | {incident_type}\n"
            f"[View Recording]({url})"
        )
        with self._lock:
            self._queue.append({"msg": msg, "chat": CHAT_ID,        "frame": None})
            self._queue.append({"msg": msg, "chat": POLICE_CHAT_ID, "frame": None})

    # ── Background sender ─────────────────────────────────────
    def _loop(self):
        while True:
            time.sleep(0.4)
            with self._lock:
                if not self._queue:
                    continue
                item = self._queue.pop(0)
            threading.Thread(target=self._send,
                             args=(item["msg"], item["chat"], item["frame"]),
                             daemon=True).start()

    def _send(self, msg, chat_id, frame=None):
        try:
            if frame is not None:
                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                cv2.imwrite(tmp.name, frame)
                tmp.close()
                try:
                    with open(tmp.name, "rb") as p:
                        requests.post(f"{BASE}/sendPhoto",
                                      data={"chat_id": chat_id,
                                            "caption": msg,
                                            "parse_mode": "Markdown"},
                                      files={"photo": p}, timeout=10)
                finally:
                    os.unlink(tmp.name)
            else:
                requests.post(f"{BASE}/sendMessage",
                              data={"chat_id": chat_id,
                                    "text": msg,
                                    "parse_mode": "Markdown"},
                              timeout=10)
        except Exception as e:
            print(f"[Telegram] Failed: {e}")


# ── Singleton ─────────────────────────────────────────────────
_bot = None
def get_bot():
    global _bot
    if _bot is None:
        _bot = TelegramAlert()
    return _bot
