"""
hud_renderer.py
HUD, bounding boxes, alerts, FPS, debug panel.
Press S to toggle debug panel showing LSTM probability bars.
"""

import cv2, time
import numpy as np
from typing import List, Optional
from detection_engine import PersonState, Alert, Config

RED    = (0,   0,   255)
ORANGE = (0,   140, 255)
YELLOW = (0,   220, 220)
GREEN  = (0,   200,  80)
WHITE  = (255, 255, 255)
BLACK  = (0,   0,     0)
GREY   = (140, 140, 140)
DARK   = (20,  20,   20)

SEV_COL   = {"critical": RED, "warning": ORANGE, "info": YELLOW}
LSTM_COL  = {"Normal": GREEN, "Fighting": RED, "Person Down": ORANGE}
FONT      = cv2.FONT_HERSHEY_SIMPLEX


def _txt(f, t, pos, col=WHITE, sc=0.55, th=2):
    cv2.putText(f, t, pos, FONT, sc, BLACK, th+2)
    cv2.putText(f, t, pos, FONT, sc, col,   th)


def _box(f, x1, y1, x2, y2, col, alpha=0.55):
    y1=max(0,y1); y2=min(f.shape[0],y2)
    x1=max(0,x1); x2=min(f.shape[1],x2)
    if x2<=x1 or y2<=y1: return
    sub=f[y1:y2,x1:x2]
    cv2.addWeighted(np.full_like(sub,col),alpha,sub,1-alpha,0,sub)
    f[y1:y2,x1:x2]=sub


class HUDRenderer:
    def __init__(self):
        self._fps_buf = []

    def draw(self, frame, persons, g_alerts, risk,
             recording, upload_url=None, show_debug=False, lstm=None):
        h, w      = frame.shape[:2]
        emergency = risk >= Config.EMERGENCY_THRESH

        self._bar(frame, w, risk, emergency, recording)
        self._boxes(frame, persons)
        self._global(frame, w, g_alerts)
        self._fps(frame, w, h)
        if emergency:
            self._flash(frame, w, h)
        if upload_url:
            _txt(frame, f"Drive:{upload_url[:46]}", (8,h-10), GREEN, 0.37, 1)
        if show_debug and persons:
            self._debug(frame, persons)
        return frame

    def _bar(self, frame, w, risk, emergency, recording):
        _box(frame, 0, 0, w, 52, DARK, 0.80)
        col = RED if emergency else GREEN
        _txt(frame, f"RISK: {risk}", (10, 36), col, 0.80, 2)

        x0,y0,y1 = 130,12,40
        x1 = w-160
        fw = int(min(risk/80,1.0)*(x1-x0))
        cv2.rectangle(frame,(x0,y0),(x1,y1),(50,50,50),-1)
        if fw>0: cv2.rectangle(frame,(x0,y0),(x0+fw,y1),col,-1)
        cv2.rectangle(frame,(x0,y0),(x1,y1),(100,100,100),1)

        st = "!! EMERGENCY !!" if emergency else "MONITORING"
        _txt(frame, st, (x1+8,36), col, 0.57)

        if recording and int(time.time()*2)%2==0:
            cv2.circle(frame,(w-18,26),8,RED,-1)
            _txt(frame,"REC",(w-14,30),WHITE,0.33,1)

    def _boxes(self, frame, persons):
        for s in persons:
            x1,y1,x2,y2 = s.box
            sevs  = [a.severity for a in s.alerts]
            col   = RED if "critical" in sevs else ORANGE if "warning" in sevs else GREEN
            thick = 3 if s.is_lying else 2
            cv2.rectangle(frame,(x1,y1),(x2,y2),col,thick)
            ly = y2+18
            for a in s.alerts:
                c = SEV_COL.get(a.severity, WHITE)
                _txt(frame, a.text, (x1,ly), c, 0.47, 1)
                ly += 18
            st = f"tilt:{s.tilt:.0f} wv:{s.wrist_vel:.0f} tv:{s.torso_vel:.0f}"
            _txt(frame, st, (x1,y1-6), GREY, 0.35, 1)

    def _global(self, frame, w, alerts):
        seen,uniq=[],[]
        for a in alerts:
            if a.text not in seen:
                seen.append(a.text); uniq.append(a)
        bx,by = w//2-200, 65
        for a in uniq[:4]:
            c = SEV_COL.get(a.severity, WHITE)
            _box(frame,bx,by-18,bx+400,by+6,c,0.18)
            _txt(frame, a.text, (bx+8,by), c, 0.65, 2)
            by+=32

    def _fps(self, frame, w, h):
        now=time.time()
        self._fps_buf.append(now)
        self._fps_buf=[t for t in self._fps_buf if now-t<1.0]
        fps=len(self._fps_buf)
        col=GREEN if fps>=15 else ORANGE if fps>=8 else RED
        _txt(frame,f"FPS:{fps}",(w-78,h-10),col,0.44,1)

    def _flash(self, frame, w, h):
        if int(time.time()*3)%2==0:
            cv2.rectangle(frame,(3,3),(w-3,h-3),RED,4)

    def _debug(self, frame, persons):
        ph = len(persons)*90+10
        _box(frame,0,52,260,52+ph,BLACK,0.75)
        for i,s in enumerate(persons):
            by = 62+i*90
            _txt(frame,f"P{s.pid}",(6,by),WHITE,0.43,1)
            # Buffer bar
            fw=int((s.buf_size/30)*120)
            cv2.rectangle(frame,(6,by+6),(126,by+16),(60,60,60),-1)
            cv2.rectangle(frame,(6,by+6),(6+fw,by+16),YELLOW,-1)
            _txt(frame,f"{s.buf_size}/30",(130,by+15),GREY,0.34,1)
            # Prob bars
            for k,cls in enumerate(["Normal","Fighting","Person Down"]):
                p   = s.lstm_probs.get(cls,0.0)
                bw2 = int(p*120)
                y   = by+22+k*18
                col = LSTM_COL.get(cls,GREY)
                cv2.rectangle(frame,(6,y),(126,y+12),(50,50,50),-1)
                if bw2>0: cv2.rectangle(frame,(6,y),(6+bw2,y+12),col,-1)
                _txt(frame,f"{cls[:8]:<8} {p:.0%}",(130,y+10),col,0.34,1)
            vc = LSTM_COL.get(s.lstm_label,GREY)
            _txt(frame,f"→{s.lstm_label}",(6,by+78),vc,0.41,1)
