# Smart-Attendance
# Smart Attendance System with CNN Anti-Spoofing

1. This project is a smart attendance system using face verification.
2. It is developed using Python.
3. FastAPI is used to build the backend.
4. OpenCV is used for face detection and image processing.
5. NumPy is used for numerical operations and face descriptors.
6. MediaPipe is used for facial landmark analysis.
7. PyTorch is used for CNN-based anti-spoofing.
8. SQLite is used to store attendance records.
9. HTML, CSS, and JavaScript provide the web interface.
10. Students can enroll using their student ID and face image.
11. The system compares the current face with enrolled data.
12. Challenge-response checks evaluate facial movements.
13. The CNN estimates the probability of a spoof attack.
14. Attendance is accepted only when required checks pass.
15. Verification attempts are stored in the database.
16. The interface displays verification results and attendance history.
17. Install dependencies using `pip install -r requirements.txt`.
18. Run the app using `python -m uvicorn app.main:app --reload`.
19. Open `http://127.0.0.1:8000` in your browser.

