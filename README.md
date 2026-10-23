# Smart Attendance — revised runnable prototype

This package is a **starter prototype**, not a validated biometric security system. It provides a FastAPI browser-camera UI, student enrollment baseline, challenge prompt endpoint, SQLite attempt logging, and a clearly labelled image-texture heuristic. It does **not** yet contain a trained anti-spoof CNN or automated facial-landmark challenge validation; the UI calls these limitations out rather than presenting them as completed features.

## Windows setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
.\run.ps1
```

Open http://127.0.0.1:8000. Allow camera access. Enroll a test ID by capturing a face; verify using the same ID. This prototype uses a grayscale histogram descriptor and is **not suitable for real attendance deployment or identity-security claims**.

## What is implemented
- FastAPI web app with browser camera capture.
- Face detection using OpenCV Haar cascade; exactly one face is required.
- Local student descriptor enrollment.
- Random challenge prompt and a manual demo checkbox (not automated liveness proof).
- A heuristic texture risk score (not a CNN probability).
- SQLite attempt log and recent-attempt table.

## Next research steps
1. Replace the histogram descriptor with a validated face embedding model and evaluate false matches.
2. Implement MediaPipe landmark-based blink/head-pose/smile/nod verification with randomized challenge ordering and server-side timing.
3. Train a CNN using bona fide, print-attack and replay-attack data; keep person/session-separated train/validation/test splits.
4. Calibrate thresholds on validation data, then report FAR, FRR, HTER and per-attack results on a held-out test set.
5. Add duplicate-attendance policy, consent/privacy controls, data deletion and access controls before any real deployment.

Never use the current heuristic score as a CNN result. Never report expected or synthetic results as measured real-world performance.


## Automatic liveness prototype update

The Verify button now captures a short sequence of webcam frames and uses MediaPipe Face Mesh landmarks to check the requested blink, head-turn, smile, or nod challenge. The UI checkbox is no longer used as proof of liveness. Click **Get random challenge**, read the prompt, then click **Capture challenge & verify** and perform the action while frames are collected.

This is a classroom research prototype, not a security guarantee. Landmark thresholds vary with lighting, camera angle, and face tracking quality. The texture score remains a heuristic and is not a trained anti-spoofing CNN. Do not report it as CNN accuracy.
