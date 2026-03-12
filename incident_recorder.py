"""
incident_recorder.py
Records emergency clips and uploads to Google Drive.
"""

import cv2, os, time, threading, pickle
from datetime import datetime
from pathlib import Path

try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    DRIVE_OK = True
except ImportError:
    DRIVE_OK = False

SCOPES   = ["https://www.googleapis.com/auth/drive.file"]
CREDS    = "credentials.json"
TOKEN    = "token.pickle"
SAVE_DIR = Path("incidents")
SAVE_DIR.mkdir(exist_ok=True)


def _drive_service():
    if not DRIVE_OK:
        return None
    creds = None
    if os.path.exists(TOKEN):
        with open(TOKEN,"rb") as f: creds=pickle.load(f)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        elif os.path.exists(CREDS):
            flow  = InstalledAppFlow.from_client_secrets_file(CREDS,SCOPES)
            creds = flow.run_local_server(port=0)
        else:
            return None
        with open(TOKEN,"wb") as f: pickle.dump(creds,f)
    return build("drive","v3",credentials=creds)


def _upload(path, svc):
    if not svc: return None
    try:
        name  = os.path.basename(path)
        meta  = {"name":name,"mimeType":"video/avi"}
        media = MediaFileUpload(path,mimetype="video/avi",resumable=True)
        fid   = svc.files().create(body=meta,media_body=media,
                                   fields="id").execute().get("id")
        svc.permissions().create(fileId=fid,
                                  body={"type":"anyone","role":"reader"}).execute()
        url = f"https://drive.google.com/file/d/{fid}/view"
        return url
    except Exception as e:
        print(f"[Drive] Upload failed: {e}")
        return None


class IncidentRecorder:
    def __init__(self, fps=20, size=(640,480), max_sec=30):
        self.fps=fps; self.size=size; self.max_sec=max_sec
        self._writer=None; self._path=None; self._t0=None
        self.is_recording=False
        self._svc=_drive_service()
        self.last_url=None

    def start(self, label="incident"):
        if self.is_recording: return
        ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe       = label.replace(" ","_").replace("/","-")[:30]
        self._path = str(SAVE_DIR/f"{safe}_{ts}.avi")
        self._writer = cv2.VideoWriter(
            self._path, cv2.VideoWriter_fourcc(*"XVID"), self.fps, self.size
        )
        self._t0=time.time(); self.is_recording=True
        print(f"[Rec] → {self._path}")

    def write(self, frame):
        if not self.is_recording: return
        if time.time()-self._t0>self.max_sec: self.stop(); return
        self._writer.write(cv2.resize(frame,self.size))

    def stop(self):
        if not self.is_recording: return
        self._writer.release(); self._writer=None
        self.is_recording=False; path=self._path
        print(f"[Rec] Saved → {path}")
        threading.Thread(target=lambda:self._upload_bg(path),daemon=True).start()

    def _upload_bg(self, path):
        url=_upload(path,self._svc)
        if url: self.last_url=url; print(f"[Drive] {url}")
