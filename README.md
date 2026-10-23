# Smart Attendance



## Windows setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
.\run.ps1
```

Open http://127.0.0.1:8000. Allow camera access. Enroll a test ID by capturing a face; verify using the same ID. This uses a grayscale histogram descriptor 

## What is implemented
- FastAPI web app with browser camera capture.
- Face detection using OpenCV Haar cascade; exactly one face is required.
- Local student descriptor enrollment.
- Random challenge prompt and a manual demo checkbox (not automated liveness proof).
- A heuristic texture risk score (not a CNN probability).
- SQLite attempt log and recent-attempt table.



## Automatic liveness prototype update

The Verify button now captures a short sequence of webcam frames and uses MediaPipe Face Mesh landmarks to check the requested blink, head-turn, smile, or nod challenge. The UI checkbox is no longer used as proof of liveness. Click **Get random challenge**, read the prompt, then click **Capture challenge & verify** and perform the action while frames are collected.


