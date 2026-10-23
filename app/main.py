
from __future__ import annotations

import base64
import json
import random
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.antispoof_cnn import predict_antispoof

try:
    import mediapipe as mp
except Exception:
    mp = None


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(
    (ROOT / "config.yaml").read_text(encoding="utf-8")
)
DB_PATH = ROOT / CONFIG.get("attendance", {}).get(
    "db_path", "attendance.db"
)
ENROLLED = ROOT / "data" / "enrolled"
ENROLLED.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Smart Attendance - CNN Anti-Spoofing",
    version="1.1.0",
)

CHALLENGES = [
    "Blink twice",
    "Turn your head left",
    "Turn your head right",
    "Smile",
    "Nod your head",
]


class ImageRequest(BaseModel):
    student_id: str = Field(min_length=1, max_length=50)
    image: str


class VerifyRequest(BaseModel):
    student_id: str = Field(min_length=1, max_length=50)
    mode: str = "full"
    challenge: str
    image: str
    frames: list[str] = Field(default_factory=list, max_length=30)
    challenge_passed: bool = False


def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            mode TEXT NOT NULL,
            recognized INTEGER NOT NULL,
            liveness_ok INTEGER NOT NULL,
            spoof_score REAL NOT NULL,
            challenges TEXT,
            reason TEXT
        )
    """)
    con.commit()
    return con


def decode_image(data: str) -> np.ndarray:
    try:
        payload = data.split(",", 1)[-1]
        raw = base64.b64decode(payload, validate=True)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Invalid image")

        return img
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not decode the camera image.",
        ) from exc


def face_crop(img: np.ndarray) -> Optional[np.ndarray]:
    if img is None or not isinstance(img, np.ndarray):
        return None

    if img.size == 0 or len(img.shape) != 3:
        return None

    if img.shape[2] != 3:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )
    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError("Could not load OpenCV face detector.")

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(60, 60),
    )

    # Require exactly one detected face.
    if len(faces) != 1:
        return None

    x, y, w, h = faces[0]
    crop = img[y:y + h, x:x + w]

    if crop.size == 0:
        return None

    return cv2.resize(crop, (128, 128))


def descriptor(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    return (
        cv2.calcHist(
            [gray], [0], None, [64], [0, 256]
        )
        .flatten()
        .astype("float32")
        / max(crop.shape[0] * crop.shape[1], 1)
    )


def recognize(
    student_id: str, crop: np.ndarray
) -> tuple[bool, str]:
    path = ENROLLED / f"{student_id}.npz"

    if not path.exists():
        return False, (
            f"Student ID '{student_id}' is not enrolled. "
            "Enroll this ID before verification."
        )

    try:
        with np.load(path) as data:
            if "descriptor" not in data:
                return False, "Enrollment file has no descriptor."

            saved = data["descriptor"].astype("float32")

        current = descriptor(crop).astype("float32")

        if saved.shape != current.shape:
            return False, "Enrollment descriptor shape mismatch."

        distance = cv2.compareHist(
            saved,
            current,
            cv2.HISTCMP_BHATTACHARYYA,
        )

        if not np.isfinite(distance):
            return False, "Recognition distance was invalid."

        matched = distance < 0.22

        return matched, (
            f"Histogram distance: {distance:.3f}; "
            f"threshold: 0.220; "
            f"{'matched' if matched else 'no match'}."
        )

    except Exception as exc:
        return False, (
            f"Could not compare enrollment: "
            f"{type(exc).__name__}: {exc}"
        )


def _landmark_metrics(img: np.ndarray, mesh):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    pts = result.multi_face_landmarks[0].landmark

    def xy(i):
        return np.array([pts[i].x, pts[i].y], dtype=float)

    def dist(a, b):
        return float(np.linalg.norm(a - b))

    def ear(ids):
        p1, p2, p3, p4, p5, p6 = [xy(i) for i in ids]
        return (
            dist(p2, p6) + dist(p3, p5)
        ) / (2 * dist(p1, p4) + 1e-6)

    left_ear = ear([33, 160, 158, 133, 153, 144])
    right_ear = ear([362, 385, 387, 263, 373, 380])

    eye_mid = (xy(33) + xy(263)) / 2
    nose = xy(1)
    face_width = max(dist(xy(234), xy(454)), 1e-6)

    yaw = float((nose[0] - eye_mid[0]) / face_width)
    smile = dist(xy(61), xy(291)) / face_width
    eye_line = (xy(159)[1] + xy(386)[1]) / 2
    pitch = float((nose[1] - eye_line) / face_width)

    return {
        "ear": (left_ear + right_ear) / 2,
        "yaw": yaw,
        "smile": smile,
        "pitch": pitch,
    }


def validate_liveness(
    challenge: str, frame_data: list[str]
) -> tuple[bool, str]:
    if len(frame_data) < 5:
        return False, "Not enough frames. Please try again."

    if mp is None:
        raise HTTPException(
            status_code=500,
            detail="MediaPipe is unavailable. Check requirements.txt.",
        )

    metrics = []
    picks = np.linspace(
        0, len(frame_data) - 1,
        min(18, len(frame_data)),
        dtype=int,
    )

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as mesh:
        for idx in picks:
            metric = _landmark_metrics(
                decode_image(frame_data[int(idx)]), mesh
            )
            if metric is not None:
                metrics.append(metric)

    if len(metrics) < max(4, int(len(picks) * 0.55)):
        return False, (
            "Could not track your face through enough frames. "
            "Improve lighting and retry."
        )

    ears = np.array([m["ear"] for m in metrics])
    yaws = np.array([m["yaw"] for m in metrics])
    smiles = np.array([m["smile"] for m in metrics])
    pitches = np.array([m["pitch"] for m in metrics])

    challenge = challenge.strip().lower()

    if "blink" in challenge:
        closed = ears < 0.19
        count = sum(
            bool(closed[i] and (i == 0 or not closed[i - 1]))
            for i in range(len(closed))
        )
        passed = count >= (2 if "twice" in challenge else 1)
        return passed, f"Blink check: {count} blink(s) detected."

    if "left" in challenge or "right" in challenge:
        passed = float(np.ptp(yaws)) > 0.075 and (
            float(np.min(yaws)) < -0.025
            or float(np.max(yaws)) > 0.025
        )
        return passed, (
            "Head movement detected."
            if passed else "Not enough head movement detected."
        )

    if "smile" in challenge:
        passed = (
            float(np.max(smiles) - np.min(smiles)) > 0.025
            or float(np.max(smiles)) > 0.38
        )
        return passed, (
            "Smile movement detected."
            if passed else "Not enough smile movement detected."
        )

    if "nod" in challenge:
        passed = float(np.ptp(pitches)) > 0.045
        return passed, (
            "Nod movement detected."
            if passed else "Not enough nod movement detected."
        )

    return False, "Unknown challenge. Please request another challenge."


def texture_score(crop: np.ndarray) -> float:
    """Return the CNN's predicted spoof probability."""
    result = predict_antispoof(crop)
    return float(result["spoof_probability"])


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(PAGE)


@app.get("/api/health")
def health():
    con = db()
    con.close()

    try:
        from app.antispoof_cnn import load_model
        load_model()
        model_status = "loaded"
    except Exception as exc:
        model_status = f"error: {type(exc).__name__}: {exc}"

    return {
        "status": "ok",
        "recognition": "histogram baseline",
        "antispoof": "trained CNN",
        "antispoof_model_status": model_status,
    }


@app.post("/api/enroll")
def enroll(req: ImageRequest):
    student_id = req.student_id.strip()
    allowed = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    )

    if not student_id or any(ch not in allowed for ch in student_id):
        raise HTTPException(
            status_code=400,
            detail="Use letters, numbers, underscore or hyphen for Student ID.",
        )

    crop = face_crop(decode_image(req.image))

    if crop is None:
        raise HTTPException(
            status_code=400,
            detail="Exactly one face must be visible. Improve lighting.",
        )

    path = ENROLLED / f"{student_id}.npz"
    desc = descriptor(crop)

    if path.exists():
        with np.load(path) as old_data:
            old = old_data["descriptor"].copy()
        desc = (old + desc) / 2.0

    np.savez_compressed(path, descriptor=desc)

    return {
        "ok": True,
        "message": f"Enrollment sample saved for {student_id}.",
    }


@app.get("/api/challenge")
def challenge():
    return {
        "challenge": random.choice(CHALLENGES),
        "note": (
            "Keep your face visible while the app captures frames. "
            "Liveness is checked using facial landmarks."
        ),
    }


@app.post("/api/verify")
def verify(req: VerifyRequest):
    mode = req.mode if req.mode in (
        "full", "blink-only", "cnn-only"
    ) else "full"

    img = decode_image(req.image)
    crop = face_crop(img)

    reason = ""
    recognized = False
    score = 1.0
    live_ok = False
    live_reason = "Liveness was not checked."

    # Uses the existing config threshold. It must be calibrated on validation data.
    threshold = float(
        CONFIG.get("antispoof", {}).get("texture_threshold", 0.55)
    )

    if crop is None:
        reason = "Exactly one face was not detected."
    else:
        recognized, rec_reason = recognize(
            req.student_id.strip(), crop
        )

        # Fail closed if the CNN cannot run.
        try:
            score = texture_score(crop)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Anti-spoof CNN failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            ) from exc

        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise HTTPException(
                status_code=503,
                detail="CNN returned an invalid spoof probability.",
            )

        if mode == "cnn-only":
            # Diagnostic mode only; never approves attendance.
            live_ok = False
            live_reason = "Liveness skipped in CNN-only diagnostic mode."
        else:
            live_ok, live_reason = validate_liveness(
                req.challenge, req.frames
            )

        if not recognized:
            reason = rec_reason
        elif score >= threshold:
            reason = (
                f"CNN flagged a possible spoof: {score:.3f} "
                f">= threshold {threshold:.3f}."
            )
        elif mode == "cnn-only":
            reason = (
                "CNN score is below threshold, but liveness was skipped. "
                "Diagnostic only; attendance was not accepted."
            )
        elif not live_ok:
            reason = live_reason
        else:
            reason = (
                f"{live_reason} CNN spoof check passed. "
            )

    
    accepted = bool(
        crop is not None
        and recognized
        and live_ok
        and score < threshold
        and mode != "cnn-only"
    )

    con = db()
    try:
        con.execute(
            """
            INSERT INTO attendance
            (student_id, timestamp, mode, recognized, liveness_ok,
             spoof_score, challenges, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                req.student_id.strip(),
                datetime.now(timezone.utc).isoformat(),
                mode,
                int(recognized),
                int(live_ok),
                score,
                json.dumps([req.challenge]),
                reason,
            ),
        )
        con.commit()
    finally:
        con.close()

    return {
        "accepted": accepted,
        "recognized": recognized,
        "liveness_ok": live_ok,
        "spoof_score": round(score, 3),
        "score_type": "CNN predicted spoof probability",
        "threshold": threshold,
        "reason": reason,
    }


@app.get("/api/attendance")
def attendance():
    con = db()
    try:
        rows = [
            dict(row)
            for row in con.execute(
                """
                SELECT id, student_id, timestamp, mode, recognized,
                       liveness_ok, spoof_score, reason
                FROM attendance
                ORDER BY id DESC
                LIMIT 100
                """
            )
        ]
    finally:
        con.close()

    return {"rows": rows}


PAGE = r'''<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Smart Attendance</title>
<style>
:root{font-family:Inter,Segoe UI,Arial,sans-serif;color:#172033;background:#f3f6fb}
body{margin:0}
.wrap{max-width:1050px;margin:auto;padding:24px}
.hero{background:#16213e;color:white;padding:24px;border-radius:18px}
.hero h1{margin:0 0 8px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}
.card{background:white;border:1px solid #dfe6f0;border-radius:16px;padding:18px;box-shadow:0 4px 14px #15213a0a}
video{width:100%;max-height:360px;background:#111;border-radius:12px;object-fit:cover}
input,select,button{font:inherit;padding:11px;border:1px solid #cbd5e1;border-radius:9px;margin:5px 0;max-width:100%}
button{background:#2457d6;color:white;border:0;cursor:pointer;font-weight:600}
button.secondary{background:#e8eefc;color:#193b91}
button:disabled{opacity:.5}
small,.muted{color:#65728a}
.status{padding:12px;border-radius:10px;background:#f0f4fa;margin-top:10px;white-space:pre-wrap}
.ok{background:#e4f8ec}
.bad{background:#fff0ed}
table{width:100%;border-collapse:collapse;font-size:13px}
td,th{text-align:left;padding:8px;border-bottom:1px solid #e5eaf2;overflow-wrap:anywhere}
@media(max-width:720px){.grid{grid-template-columns:1fr}.wrap{padding:12px}}
</style>
</head>
<body>
<div class="wrap">
<header class="hero">
<h1>Smart Attendance</h1>
<div>Face enrollment · challenge-response · CNN anti-spoofing</div>
</header>

<div class="grid">
<section class="card">
<h2>Camera</h2>
<video id="video" autoplay playsinline muted></video>
<canvas id="canvas" hidden></canvas>
<div>
<button id="start">Start camera</button>
<button class="secondary" id="capture">Capture frame</button>
</div>
<div id="cameraStatus" class="status">Start camera to begin.</div>
</section>

<section class="card">
<h2>1. Enroll student</h2>
<label for="sid">Student ID</label><br>
<input id="sid" placeholder="e.g. STU001" autocomplete="off"><br>
<button id="enroll">Save enrollment sample</button>
<hr>
<h2>2. Verify attendance</h2>
<label for="mode">Verification mode</label><br>
<select id="mode">
<option value="full">Full (challenge + CNN)</option>
<option value="blink-only">Blink-only + CNN</option>
<option value="cnn-only">CNN-only diagnostic (no attendance)</option>
</select>
<p id="challenge">Press “Get random challenge”.</p>
<button class="secondary" id="getChallenge">Get random challenge</button><br>
<p id="captureStatus" class="muted">Verification captures frames and checks facial landmarks automatically.</p>
<button id="verify">Capture challenge &amp; verify</button>
<div id="result" class="status">No verification yet.</div>
</section>
</div>

<section class="card" style="margin-top:16px">
<h2>Recent attempts</h2>
<button class="secondary" id="refresh">Refresh attendance log</button>
<div style="overflow:auto;margin-top:10px">
<table>
<thead><tr><th>Student</th><th>Time (UTC)</th><th>Mode</th><th>Recognized</th><th>Liveness</th><th>CNN spoof score</th><th>Reason</th></tr></thead>
<tbody id="rows"></tbody>
</table>
</div>
</section>
</div>

<script>
let stream=null,lastImage=null,currentChallenge='';
const $=id=>document.getElementById(id);

async function start(){
  try{
    stream=await navigator.mediaDevices.getUserMedia({
      video:{facingMode:'user'},audio:false
    });
    $('video').srcObject=stream;
    $('cameraStatus').textContent='Camera ready.';
  }catch(e){
    $('cameraStatus').textContent='Camera error: '+e.message+
      ' — allow camera permission and use localhost.';
  }
}

function capture(){
  const v=$('video'),c=$('canvas');
  if(!v.videoWidth){alert('Start the camera first.');return null;}
  c.width=v.videoWidth;c.height=v.videoHeight;
  c.getContext('2d').drawImage(v,0,0);
  lastImage=c.toDataURL('image/jpeg',.9);
  $('cameraStatus').textContent='Frame captured.';
  return lastImage;
}

async function post(url,data){
  const r=await fetch(url,{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify(data)
  });
  const text=await r.text();
  let d;
  try{d=JSON.parse(text)}
  catch{
    throw Error('Server returned non-JSON (HTTP '+r.status+').');
  }
  if(!r.ok)throw Error((d&&d.detail)||('Request failed (HTTP '+r.status+')'));
  return d;
}

$('start').onclick=start;
$('capture').onclick=capture;

$('enroll').onclick=async()=>{
  try{
    const image=capture();if(!image)return;
    const d=await post('/api/enroll',{
      student_id:$('sid').value,image
    });
    $('result').className='status ok';
    $('result').textContent=d.message;
  }catch(e){
    $('result').className='status bad';
    $('result').textContent=e.message;
  }
};

$('getChallenge').onclick=async()=>{
  try{
    const r=await fetch('/api/challenge');
    const d=await r.json();
    currentChallenge=d.challenge;
    $('challenge').textContent='Challenge: '+d.challenge+' — '+d.note;
  }catch(e){
    $('challenge').textContent=e.message;
  }
};

async function captureSequence(durationMs=3600,intervalMs=200){
  const v=$('video'),c=$('canvas');
  if(!v.videoWidth)throw Error('Start the camera first.');
  const frames=[],end=Date.now()+durationMs;
  $('captureStatus').textContent='Capturing challenge frames — perform the action now…';
  while(Date.now()<end){
    c.width=v.videoWidth;c.height=v.videoHeight;
    c.getContext('2d').drawImage(v,0,0);
    frames.push(c.toDataURL('image/jpeg',.75));
    await new Promise(r=>setTimeout(r,intervalMs));
  }
  lastImage=frames[frames.length-1];
  $('captureStatus').textContent='Captured '+frames.length+' frames.';
  return frames;
}

$('verify').onclick=async()=>{
  try{
    if(!currentChallenge)throw Error('Click “Get random challenge” first.');
    if(!$('sid').value.trim())throw Error('Enter the enrolled Student ID first.');

    const frames=await captureSequence();
    const d=await post('/api/verify',{
      student_id:$('sid').value,
      mode:$('mode').value,
      challenge:currentChallenge,
      image:frames[frames.length-1],
      frames
    });

    $('result').className='status '+(d.accepted?'ok':'bad');
    $('result').textContent=
      (d.accepted?'ATTEMPT ACCEPTED ':'ATTEMPT REJECTED')+
      '\nRecognized: '+d.recognized+
      '\nAutomatic liveness: '+d.liveness_ok+
      '\nCNN spoof probability: '+d.spoof_score+
      ' (threshold: '+d.threshold+')\n'+d.reason;

    $('captureStatus').textContent='Sequence analysis complete.';
    await refresh();
  }catch(e){
    $('result').className='status bad';
    $('result').textContent=e.message;
    $('captureStatus').textContent='Capture or verification failed.';
  }
};

async function refresh(){
  try{
    const r=await fetch('/api/attendance');
    const d=await r.json();
    $('rows').innerHTML=d.rows.map(r=>
      '<tr><td>'+esc(r.student_id)+'</td><td>'+esc(r.timestamp)+
      '</td><td>'+esc(r.mode)+'</td><td>'+r.recognized+
      '</td><td>'+r.liveness_ok+'</td><td>'+
      Number(r.spoof_score).toFixed(3)+'</td><td>'+
      esc(r.reason||'')+'</td></tr>'
    ).join('');
  }catch(e){
    $('rows').innerHTML='<tr><td colspan="7">'+esc(e.message)+'</td></tr>';
  }
}

function esc(s){
  return String(s).replace(/[&<>"']/g,c=>({
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
  }[c]));
}

$('refresh').onclick=refresh;
refresh();
</script>
</body>
</html>'''
