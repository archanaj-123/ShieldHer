"""
detection_engine.py
──────────────────────────────────────────────────────────────────
False positive fixes vs previous version:

PROBLEM 1: Two people near each other triggered assault instantly
FIX: Assault now needs BOTH fast wrist AND close proximity
     AND sustained for multiple frames (new assault_count tracker)
     ASSAULT_DIST raised 220 → 140px (must be really close)
     WRIST_VEL raised 60 → 90px/frame (must be very fast)

PROBLEM 2: Normal walking/movement raised risk
FIX: TORSO_VEL raised 50 → 80, shove also needs proximity + sustained

PROBLEM 3: Sitting/leaning triggered lying detection
FIX: LYING_RATIO raised 1.25 → 1.6 (must be very wide box)
     LYING_CONFIRM raised 5 → 10 frames

PROBLEM 4: Risk window too long — old events kept adding up
FIX: RISK_WINDOW reduced 20 → 15, EMERGENCY_THRESH raised 50 → 60

PROBLEM 5: LSTM firing too easily
FIX: VOTE_NEEDED raised, confidence threshold raised to 0.70
──────────────────────────────────────────────────────────────────
"""

import cv2
import math
import time
import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from lstm_classifier import LSTMClassifier


class Config:
    POSE_MODEL = "yolov8n-pose.pt"
    OBJ_MODEL  = "yolov8n.pt"
    POSE_CONF  = 0.35
    OBJ_CONF   = 0.30   # lower = detects more objects
    LSTM_MODEL = "pose_lstm.pt"
    W, H       = 640, 480

    EMERGENCY_THRESH = 60    # raised — harder to trigger
    RISK_WINDOW      = 15    # shorter window — old events fade faster

    # Distress — hands raised
    HANDS_UP_SECS = 4.0      # must hold for 4 full seconds
    HAND_UP_PX    = 35       # wrist must be clearly above shoulder

    # Fall / faint detection
    TILT_DEG      = 55       # degrees from vertical (higher = less sensitive)
    TILT_CONFIRM  = 10       # must be tilted for 10 consecutive frames
    LYING_RATIO   = 1.6      # box width/height — must be clearly wide
    LYING_CONFIRM = 10       # 10 frames of lying aspect ratio

    # Speed thresholds
    STILL_VEL     = 6        # px/frame below = person is still

    # Assault — MUCH stricter now
    WRIST_VEL       = 90    # px/frame — very fast arm movement
    TORSO_VEL       = 80    # px/frame — very fast body movement
    ASSAULT_DIST    = 140   # px — must be very close (not just in frame)
    ASSAULT_CONFIRM = 5     # must be sustained for 5 frames

    # Crowd
    CROWD_N    = 5
    CROWD_DIST = 120

    # ── Harassment detection ──────────────────────────────────
    # 1. Prolonged proximity — too close for too long
    HARASS_CLOSE_DIST   = 160   # px — personal space violation
    HARASS_CLOSE_SECS   = 8.0   # seconds sustained closeness = alert
    # 2. Blocking — victim barely moving while aggressor stays close
    HARASS_BLOCK_VEL    = 5     # px/frame — victim considered blocked/still
    HARASS_BLOCK_SECS   = 6.0   # seconds of being blocked
    # 3. Grab / touch — aggressor wrist near victim torso
    HARASS_GRAB_DIST    = 75    # px — wrist of A near body centre of B
    HARASS_GRAB_SECS    = 2.0   # seconds wrist stays near body
    # 4. Cornering — victim near frame edge, aggressor close
    HARASS_CORNER_EDGE  = 80    # px from frame edge = cornered
    # 5. Following — person mirrors another's movement direction
    HARASS_FOLLOW_SECS  = 10.0  # seconds of sustained following
    HARASS_FOLLOW_DIST  = 200   # px — must be close enough to be following
    # 6. Repeated approach — person keeps re-approaching after moving away
    HARASS_APPROACH_CNT = 3     # number of approaches = pattern alert
    HARASS_APPROACH_GAP = 4.0   # seconds gap between approaches to count as new

    # Min box — ignore partial/small detections
    MIN_H = 100
    MIN_W = 50

    # ── Threat objects — detected by YOLOv8 object model ─────
    # These are actual COCO class names YOLOv8 can detect.
    # Risk points assigned per object type.
    THREAT_OBJECTS = {
        # Weapons
        "knife":        10,
        "scissors":      6,
        "baseball bat":  8,
        "sports ball":   2,   # can be used as projectile
        # Potentially threatening
        "bottle":        4,
        "fork":          3,
        "spoon":         2,
        "cup":           2,
        "bowl":          2,
        "vase":          2,
        "remote":        1,
        "cell phone":    1,   # not a weapon but suspicious in assault context
    }

    # ── Suspicious objects — shown on screen but lower risk ──
    SUSPICIOUS_OBJECTS = {
        "cell phone": 1,
        "laptop":     1,
        "backpack":   1,
        "handbag":    1,
        "suitcase":   1,
        "umbrella":   2,
    }

    # ── All COCO class names YOLOv8n.pt can detect ───────────
    # (80 classes — full list for reference)
    # We show ALL detections on screen, only score the ones above

    RISK = {
        "distress":         6,
        "fall_still":       8,
        "fall_move":        4,
        "lying_still":      9,
        "lying_move":       4,
        "assault":          9,
        "shove":            5,
        "crowd":            2,
        "crowd_fall":       6,
        # Harassment
        "harass_proximity": 5,
        "harass_blocking":  6,
        "harass_grab":      8,
        "harass_cornering": 7,
        "harass_following": 5,
        "harass_pattern":   7,
    }


@dataclass
class Alert:
    text:     str
    severity: str
    risk:     int = 0


@dataclass
class PersonState:
    pid:        int
    box:        Tuple[int,int,int,int]
    centre:     Tuple[int,int]
    alerts:     List[Alert] = field(default_factory=list)
    tilt:       float = 0.0
    torso_vel:  float = 0.0
    wrist_vel:  float = 0.0
    is_lying:   bool  = False
    lstm_label: str   = "Normal"
    lstm_conf:  float = 0.0
    lstm_probs: dict  = field(default_factory=dict)
    buf_size:   int   = 0


def _dist(a, b) -> float:
    return float(np.linalg.norm(np.array(a, float) - np.array(b, float)))

def _vis(kp, thr=10) -> bool:
    """Keypoint is reliably visible."""
    return float(kp[0]) > thr and float(kp[1]) > thr

def _tilt(sh, hip) -> float:
    v = np.array(hip, float) - np.array(sh, float)
    return abs(math.degrees(math.atan2(abs(v[0]), abs(v[1]) + 1e-6)))

def _ratio(x1, y1, x2, y2) -> float:
    return max(x2-x1, 1) / max(y2-y1, 1)


class Memory:
    def __init__(self):
        self.torso_pos:     Dict[int, tuple] = {}
        self.wrist_pos:     Dict[int, dict]  = {}
        self.hands_since:   Dict[int, float] = {}
        self.tilt_cnt:      Dict[int, int]   = defaultdict(int)
        self.lying_cnt:     Dict[int, int]   = defaultdict(int)
        self.assault_cnt:   Dict[str, int]   = defaultdict(int)
        self.shove_cnt:     Dict[str, int]   = defaultdict(int)
        self.risk_hist:     deque            = deque(maxlen=Config.RISK_WINDOW)

        # ── Harassment trackers (keyed by "pidA-pidB") ────────
        # How long pair has been in close proximity
        self.close_since:   Dict[str, float] = {}
        # How long victim has been blocked (still while other is close)
        self.block_since:   Dict[str, float] = {}
        # How long aggressor wrist has been near victim body
        self.grab_since:    Dict[str, float] = {}
        # How long following behaviour has been sustained
        self.follow_since:  Dict[str, float] = {}
        # History of each person's positions for direction tracking
        self.pos_history:   Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))
        # Approach counter: how many times A has approached B
        self.approach_cnt:  Dict[str, int]   = defaultdict(int)
        self.approach_last: Dict[str, float] = defaultdict(float)
        # Was pair close last frame (to detect new approaches)
        self.was_close:     Dict[str, bool]  = defaultdict(bool)

    def torso_vel(self, pid, pos) -> float:
        v = _dist(self.torso_pos[pid], pos) if pid in self.torso_pos else 0.0
        self.torso_pos[pid] = pos
        return v

    def wrist_vel(self, pid, lw, rw):
        p  = self.wrist_pos.get(pid, {})
        lv = _dist(p.get("l", lw), lw) if _vis(lw) else 0.0
        rv = _dist(p.get("r", rw), rw) if _vis(rw) else 0.0
        self.wrist_pos[pid] = {"l": tuple(lw), "r": tuple(rw)}
        return lv, rv

    def hands_dur(self, pid, both_up, now) -> float:
        if both_up:
            self.hands_since.setdefault(pid, now)
            return now - self.hands_since[pid]
        else:
            if pid in self.hands_since and now - self.hands_since[pid] > 1.5:
                del self.hands_since[pid]
            return 0.0

    def tick_tilt(self, pid, tilted) -> int:
        self.tilt_cnt[pid] = (min(self.tilt_cnt[pid]+1, 60) if tilted
                              else max(self.tilt_cnt[pid]-2, 0))
        return self.tilt_cnt[pid]

    def tick_lying(self, pid, lying) -> int:
        self.lying_cnt[pid] = (min(self.lying_cnt[pid]+1, 60) if lying
                               else max(self.lying_cnt[pid]-2, 0))
        return self.lying_cnt[pid]

    def tick_assault(self, key, active) -> int:
        self.assault_cnt[key] = (min(self.assault_cnt[key]+1, 30) if active
                                 else max(self.assault_cnt[key]-2, 0))
        return self.assault_cnt[key]

    def tick_shove(self, key, active) -> int:
        self.shove_cnt[key] = (min(self.shove_cnt[key]+1, 30) if active
                               else max(self.shove_cnt[key]-2, 0))
        return self.shove_cnt[key]

    def push_risk(self, score) -> int:
        self.risk_hist.append(score)
        return int(sum(self.risk_hist))


class DetectionEngine:

    def __init__(self):
        from ultralytics import YOLO
        print("[Engine] Loading models...")
        self.pose = YOLO(Config.POSE_MODEL)
        self.obj  = YOLO(Config.OBJ_MODEL)
        self.lstm = LSTMClassifier(Config.LSTM_MODEL)
        self.mem  = Memory()
        print("[Engine] Ready.")

    def process(self, frame, run_obj=True):
        now      = time.time()
        risk     = 0
        g_alerts = []
        persons  = []

        if run_obj:
            risk = self._weapons(frame, risk, g_alerts)

        pose_res  = self.pose(frame, conf=Config.POSE_CONF, verbose=False)
        annotated = pose_res[0].plot(boxes=False)

        if pose_res[0].keypoints is not None:
            kps   = pose_res[0].keypoints.xy.cpu().numpy()
            boxes = pose_res[0].boxes.xyxy.cpu().numpy()
            persons, risk = self._persons(kps, boxes, now, risk, g_alerts)
            risk = self._harassment(persons, kps, boxes, now, risk, g_alerts)
            risk = self._crowd(persons, risk, g_alerts)

        smoothed = self.mem.push_risk(risk)
        return annotated, persons, g_alerts, smoothed

    # ── Object detection ──────────────────────────────────────
    def _weapons(self, frame, risk, alerts):
        res = self.obj(frame, conf=Config.OBJ_CONF, verbose=False)
        for r in res:
            for b in r.boxes:
                lbl  = self.obj.names[int(b.cls[0])].lower()
                conf = float(b.conf[0])
                x1,y1,x2,y2 = map(int, b.xyxy[0])

                # Colour based on threat level
                if lbl in Config.THREAT_OBJECTS:
                    color     = (0, 0, 255)     # red — threat
                    pts       = Config.THREAT_OBJECTS[lbl]
                    sev       = "critical" if pts >= 6 else "warning"
                    tag       = "WEAPON" if lbl in ["knife","scissors","baseball bat"] else "OBJECT"
                    alerts.append(Alert(f"{tag}: {lbl.upper()} ({conf:.0%})", sev, pts))
                    risk     += pts
                elif lbl in Config.SUSPICIOUS_OBJECTS:
                    color = (0, 165, 255)        # orange — suspicious
                    pts   = Config.SUSPICIOUS_OBJECTS[lbl]
                    risk += pts
                elif lbl == "person":
                    continue                     # persons handled by pose model
                else:
                    color = (180, 180, 180)      # grey — neutral object

                cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
                cv2.putText(frame, f"{lbl} {conf:.0%}", (x1,y1-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
        return risk

    # ── Per person ────────────────────────────────────────────
    def _persons(self, kps_all, boxes_all, now, risk, g_alerts):
        states = []

        for pid, kps in enumerate(kps_all):
            if pid >= len(boxes_all):
                continue

            x1,y1,x2,y2 = map(int, boxes_all[pid])
            bw = x2-x1; bh = y2-y1
            ar = _ratio(x1,y1,x2,y2)

            looks_lying = ar >= Config.LYING_RATIO
            if not looks_lying and (bh < Config.MIN_H or bw < Config.MIN_W):
                continue

            ls, rs = kps[5],  kps[6]
            lw, rw = kps[9],  kps[10]
            lh, rh = kps[11], kps[12]

            torso_pts = [q for q in [ls,rs,lh,rh] if _vis(q)]
            if torso_pts:
                cx,cy = np.mean(torso_pts,axis=0).astype(int)
            else:
                cx,cy = (x1+x2)//2, (y1+y2)//2
            centre = (int(cx), int(cy))

            ps = PersonState(pid=pid, box=(x1,y1,x2,y2), centre=centre)

            tv     = self.mem.torso_vel(pid, centre)
            lv, rv = self.mem.wrist_vel(pid, lw, rw)
            ps.torso_vel = tv
            ps.wrist_vel = max(lv, rv)

            body_vis = (_vis(ls) and _vis(lh)) or (_vis(rs) and _vis(rh))

            # ── CHECK 1: Lying (aspect ratio) ─────────────────
            # Person flat on floor → box is wider than tall
            lying_f = self.mem.tick_lying(pid, looks_lying)
            if lying_f >= Config.LYING_CONFIRM:
                ps.is_lying = True
                if tv < Config.STILL_VEL:
                    ps.alerts.append(Alert(
                        "PERSON DOWN — NOT MOVING",
                        "critical", Config.RISK["lying_still"]
                    ))
                    risk += Config.RISK["lying_still"]
                else:
                    ps.alerts.append(Alert(
                        "PERSON FALLING",
                        "warning", Config.RISK["lying_move"]
                    ))
                    risk += Config.RISK["lying_move"]

            # ── CHECK 2: Spine tilt (shoulder + hip visible) ───
            tilt = 0.0
            if _vis(ls) and _vis(lh):
                tilt = _tilt(ls, lh)
            elif _vis(rs) and _vis(rh):
                tilt = _tilt(rs, rh)
            ps.tilt = tilt

            tilt_f = self.mem.tick_tilt(pid, tilt > Config.TILT_DEG)
            if tilt_f >= Config.TILT_CONFIRM and not ps.is_lying:
                if tv < Config.STILL_VEL:
                    ps.alerts.append(Alert(
                        "PERSON DOWN / FAINTED",
                        "critical", Config.RISK["fall_still"]
                    ))
                    risk += Config.RISK["fall_still"]
                else:
                    ps.alerts.append(Alert(
                        "POSSIBLE FALL",
                        "warning", Config.RISK["fall_move"]
                    ))
                    risk += Config.RISK["fall_move"]

            # ── CHECK 3: Hands raised ──────────────────────────
            # Must be BOTH hands, ABOVE shoulders, for 4+ seconds
            if body_vis:
                l_up = (_vis(lw) and _vis(ls) and
                        lw[1] < ls[1] - Config.HAND_UP_PX)
                r_up = (_vis(rw) and _vis(rs) and
                        rw[1] < rs[1] - Config.HAND_UP_PX)
                dur  = self.mem.hands_dur(pid, l_up and r_up, now)
                if l_up and r_up:
                    if dur >= Config.HANDS_UP_SECS:
                        ps.alerts.append(Alert(
                            "HANDS RAISED — SOS",
                            "critical", Config.RISK["distress"]
                        ))
                        risk += Config.RISK["distress"]
                    else:
                        ps.alerts.append(Alert(
                            f"Hands Raised ({dur:.1f}s)", "info", 0
                        ))

            # ── CHECK 4: LSTM ──────────────────────────────────
            self.lstm.update(pid, kps[:17])
            lstm_risk, lstm_label, lstm_conf = self.lstm.get_risk(pid)
            ps.lstm_label = lstm_label
            ps.lstm_conf  = lstm_conf
            ps.lstm_probs = self.lstm.all_probs(pid)
            ps.buf_size   = self.lstm.buf_size(pid)

            if lstm_label != "Normal":
                sev = "critical" if lstm_risk >= 7 else "warning"
                ps.alerts.append(Alert(
                    f"[AI] {lstm_label} ({lstm_conf:.0%})",
                    sev, lstm_risk
                ))
                risk += lstm_risk

            states.append(ps)

        risk = self._assault(states, risk, g_alerts)
        return states, risk

    # ── Assault (sustained, close, fast) ──────────────────────
    def _assault(self, states, risk, g_alerts):
        """
        Only fires if:
        1. Two people are VERY close (< ASSAULT_DIST px)
        2. One person has VERY fast wrist movement (> WRIST_VEL)
        3. This is sustained for ASSAULT_CONFIRM frames
        Just being near each other does NOT trigger this.
        """
        for i in range(len(states)):
            for j in range(i+1, len(states)):
                a, b  = states[i], states[j]
                d     = _dist(a.centre, b.centre)
                key   = f"{min(a.pid,b.pid)}-{max(a.pid,b.pid)}"

                close        = d < Config.ASSAULT_DIST
                fast_wrist   = (a.wrist_vel > Config.WRIST_VEL or
                                b.wrist_vel > Config.WRIST_VEL)
                fast_torso   = (a.torso_vel > Config.TORSO_VEL or
                                b.torso_vel > Config.TORSO_VEL)

                # Assault: close + fast wrist + sustained
                assault_f = self.mem.tick_assault(key, close and fast_wrist)
                if assault_f >= Config.ASSAULT_CONFIRM:
                    g_alerts.append(Alert(
                        "ASSAULT / FIGHT DETECTED",
                        "critical", Config.RISK["assault"]
                    ))
                    risk += Config.RISK["assault"]

                # Shove: close + fast torso + sustained (but no fast wrist)
                elif not fast_wrist:
                    shove_f = self.mem.tick_shove(key, close and fast_torso)
                    if shove_f >= Config.ASSAULT_CONFIRM:
                        g_alerts.append(Alert(
                            "PUSHING / SHOVING",
                            "warning", Config.RISK["shove"]
                        ))
                        risk += Config.RISK["shove"]

        return risk

    # ── Harassment detection ──────────────────────────────────
    def _harassment(self, states, kps_all, boxes_all, now, risk, g_alerts):
        """
        Detects 6 harassment patterns between pairs of people:

        1. PROLONGED PROXIMITY  — one person stays inside personal space 8s+
        2. BLOCKING             — victim is still, aggressor stays close 6s+
        3. GRABBING / TOUCHING  — aggressor wrist near victim torso 2s+
        4. CORNERING            — victim pushed toward frame edge
        5. FOLLOWING            — person mirrors another's movement 10s+
        6. REPEATED APPROACH    — keeps coming back 3+ times

        All patterns require sustained behaviour to avoid false alarms.
        """
        n = len(states)
        if n < 2:
            # Clean up trackers if only one person left
            return risk

        # Update position history for every person
        for s in states:
            self.mem.pos_history[s.pid].append(s.centre)

        for i in range(n):
            for j in range(i + 1, n):
                a, b  = states[i], states[j]
                key   = f"{min(a.pid,b.pid)}-{max(a.pid,b.pid)}"
                d     = _dist(a.centre, b.centre)
                close = d < Config.HARASS_CLOSE_DIST

                # ── Track approach count ───────────────────────
                # Count each time pair transitions from far → close
                was_close = self.mem.was_close[key]
                if close and not was_close:
                    gap = now - self.mem.approach_last[key]
                    if gap > Config.HARASS_APPROACH_GAP:
                        self.mem.approach_cnt[key] += 1
                        self.mem.approach_last[key] = now
                self.mem.was_close[key] = close

                # ── CHECK 1: Prolonged proximity ──────────────
                if close:
                    self.mem.close_since.setdefault(key, now)
                    dur = now - self.mem.close_since[key]
                    if dur >= Config.HARASS_CLOSE_SECS:
                        g_alerts.append(Alert(
                            f"HARASSMENT — PROLONGED PROXIMITY"
                            f" P{a.pid}↔P{b.pid} ({dur:.0f}s)",
                            "warning", Config.RISK["harass_proximity"]
                        ))
                        risk += Config.RISK["harass_proximity"]
                else:
                    self.mem.close_since.pop(key, None)
                    self.mem.block_since.pop(key, None)
                    self.mem.grab_since.pop(key, None)
                    self.mem.follow_since.pop(key, None)

                if not close:
                    continue

                # ── CHECK 2: Blocking ─────────────────────────
                # One person barely moving while the other stays close
                # The still person is the victim being blocked
                a_still = a.torso_vel < Config.HARASS_BLOCK_VEL
                b_still = b.torso_vel < Config.HARASS_BLOCK_VEL
                one_still = a_still != b_still   # exactly one of them is still

                if one_still:
                    self.mem.block_since.setdefault(key, now)
                    bdur = now - self.mem.block_since[key]
                    victim = a if a_still else b
                    if bdur >= Config.HARASS_BLOCK_SECS:
                        g_alerts.append(Alert(
                            f"HARASSMENT — BLOCKING / INTIMIDATION"
                            f" (P{victim.pid} blocked {bdur:.0f}s)",
                            "warning", Config.RISK["harass_blocking"]
                        ))
                        risk += Config.RISK["harass_blocking"]
                else:
                    self.mem.block_since.pop(key, None)

                # ── CHECK 3: Grabbing / Touching ──────────────
                # Wrist of person A is very close to the torso of person B
                # Uses raw keypoints for accuracy
                grab_detected = False
                for aggressor, victim in [(a, b), (b, a)]:
                    if aggressor.pid >= len(kps_all):
                        continue
                    if victim.pid >= len(kps_all):
                        continue
                    ag_kps = kps_all[aggressor.pid]
                    vi_centre = victim.centre

                    # Check both wrists of aggressor
                    for wrist_kp in [ag_kps[9], ag_kps[10]]:
                        if not _vis(wrist_kp):
                            continue
                        wrist_pos = (float(wrist_kp[0]), float(wrist_kp[1]))
                        if _dist(wrist_pos, vi_centre) < Config.HARASS_GRAB_DIST:
                            grab_detected = True
                            break
                    if grab_detected:
                        break

                if grab_detected:
                    self.mem.grab_since.setdefault(key, now)
                    gdur = now - self.mem.grab_since[key]
                    if gdur >= Config.HARASS_GRAB_SECS:
                        g_alerts.append(Alert(
                            f"HARASSMENT — GRABBING / TOUCHING"
                            f" ({gdur:.0f}s)",
                            "critical", Config.RISK["harass_grab"]
                        ))
                        risk += Config.RISK["harass_grab"]
                else:
                    self.mem.grab_since.pop(key, None)

                # ── CHECK 4: Cornering ────────────────────────
                # Victim is near the edge of frame, aggressor is close
                for aggressor, victim in [(a, b), (b, a)]:
                    vx, vy = victim.centre
                    near_edge = (
                        vx < Config.HARASS_CORNER_EDGE or
                        vx > Config.W - Config.HARASS_CORNER_EDGE or
                        vy < Config.HARASS_CORNER_EDGE or
                        vy > Config.H - Config.HARASS_CORNER_EDGE
                    )
                    if near_edge:
                        g_alerts.append(Alert(
                            f"HARASSMENT — CORNERING"
                            f" P{victim.pid} against edge",
                            "warning", Config.RISK["harass_cornering"]
                        ))
                        risk += Config.RISK["harass_cornering"]
                        break

                # ── CHECK 5: Following ────────────────────────
                # Person B moves in same direction as person A consistently
                ph_a = list(self.mem.pos_history[a.pid])
                ph_b = list(self.mem.pos_history[b.pid])

                if len(ph_a) >= 20 and len(ph_b) >= 20:
                    # Calculate movement direction vectors (last 20 frames)
                    def movement_vec(history):
                        dx = history[-1][0] - history[-10][0]
                        dy = history[-1][1] - history[-10][1]
                        mag = math.sqrt(dx*dx + dy*dy) + 1e-6
                        return dx/mag, dy/mag

                    va = movement_vec(ph_a)
                    vb = movement_vec(ph_b)

                    # Dot product — if close to 1.0 they move same direction
                    dot = va[0]*vb[0] + va[1]*vb[1]

                    # Both must be actually moving (not standing still)
                    a_moving = _dist(ph_a[-1], ph_a[-10]) > 15
                    b_moving = _dist(ph_b[-1], ph_b[-10]) > 15

                    if dot > 0.85 and a_moving and b_moving and d < Config.HARASS_FOLLOW_DIST:
                        self.mem.follow_since.setdefault(key, now)
                        fdur = now - self.mem.follow_since[key]
                        if fdur >= Config.HARASS_FOLLOW_SECS:
                            g_alerts.append(Alert(
                                f"HARASSMENT — FOLLOWING"
                                f" P{b.pid} following P{a.pid} ({fdur:.0f}s)",
                                "warning", Config.RISK["harass_following"]
                            ))
                            risk += Config.RISK["harass_following"]
                    else:
                        self.mem.follow_since.pop(key, None)

                # ── CHECK 6: Repeated approach pattern ────────
                if self.mem.approach_cnt[key] >= Config.HARASS_APPROACH_CNT:
                    g_alerts.append(Alert(
                        f"HARASSMENT — REPEATED APPROACH PATTERN"
                        f" ({self.mem.approach_cnt[key]}x)",
                        "warning", Config.RISK["harass_pattern"]
                    ))
                    risk += Config.RISK["harass_pattern"]
                    # Reset counter so it doesn't fire every frame
                    self.mem.approach_cnt[key] = 0

        return risk

    # ── Crowd ─────────────────────────────────────────────────
    def _crowd(self, states, risk, g_alerts) -> int:
        n = len(states)
        if n < Config.CROWD_N:
            return risk
        cs    = [s.centre for s in states]
        pairs = [(i,j) for i in range(n) for j in range(i+1,n)]
        avg_d = sum(_dist(cs[i],cs[j]) for i,j in pairs) / len(pairs)
        if avg_d < Config.CROWD_DIST:
            risk += Config.RISK["crowd"]
            g_alerts.append(Alert(
                f"DENSE CROWD ({n} persons)", "warning", Config.RISK["crowd"]
            ))
            for s in states:
                if s.is_lying or s.tilt > Config.TILT_DEG:
                    risk += Config.RISK["crowd_fall"]
                    g_alerts.append(Alert(
                        "PERSON DOWN IN CROWD — CRUSH RISK",
                        "critical", Config.RISK["crowd_fall"]
                    ))
                    break
        return risk
