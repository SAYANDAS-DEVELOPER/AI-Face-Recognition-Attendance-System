import cv2
import os
import pandas as pd
from datetime import datetime, timedelta
from deepface import DeepFace
import time

# ================= PATH SETTINGS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ROUTINE_FILE = os.path.join(BASE_DIR, "CSE_class_routine.xlsx")
IMAGES_PATH = os.path.join(BASE_DIR, "images_data")
ATTENDANCE_FOLDER = os.path.join(BASE_DIR, "attendance_sheets")

START_DELAY_MIN = 5
SESSION_DURATION_MIN = 10
IDLE_SLEEP_SEC = 20

# ================= CAMERA =================
def find_camera_index():
    for idx in range(3):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.release()
            print(f"[INFO] Camera found at index {idx}")
            return idx
    return None

CAMERA_INDEX = find_camera_index()
if CAMERA_INDEX is None:
    raise SystemExit("No camera found")

# ================= STUDENTS =================
def get_all_students():
    return [os.path.splitext(f)[0] for f in os.listdir(IMAGES_PATH)]

# ================= ROUTINE =================
routine_df = pd.read_excel(ROUTINE_FILE)

def parse_time(value):
    try:
        return pd.to_datetime(value).time()
    except:
        return None

def get_today_classes():
    today = datetime.now().strftime("%A").lower()
    classes = []

    for _, row in routine_df.iterrows():
        day = str(row["Day"]).lower()
        if today in day:
            t = parse_time(row["Start Time"])
            if t:
                classes.append((row["Subject"], t))

    return sorted(classes, key=lambda x: x[1])

def get_current_class():
    now = datetime.now()
    classes = get_today_classes()

    for subject, t in classes:
        class_time = datetime.combine(now.date(), t)
        start = class_time + timedelta(minutes=START_DELAY_MIN)
        end = start + timedelta(minutes=SESSION_DURATION_MIN)

        if start <= now <= end:
            return subject

    return None

# ================= ATTENDANCE =================
def mark_attendance(subject, present_names):
    date_str = datetime.now().strftime("%Y-%m-%d")
    folder = os.path.join(ATTENDANCE_FOLDER, date_str)
    os.makedirs(folder, exist_ok=True)

    file_path = os.path.join(folder, f"{subject}.xlsx")
    all_students = get_all_students()

    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
    else:
        df = pd.DataFrame({"Name": all_students})
        df["Roll No"] = df["Name"].apply(lambda x: x.split("_")[-1])

    if date_str not in df.columns:
        df[date_str] = 0  # default Absent

    # Mark Present
    for name in present_names:
        df.loc[df["Name"] == name, date_str] = 1

    # 🔥 Attendance % Calculation
    date_columns = [col for col in df.columns if col not in ["Name", "Roll No"]]
    df["Total Present"] = df[date_columns].sum(axis=1)
    df["Attendance %"] = (df["Total Present"] / len(date_columns)) * 100

    df.to_excel(file_path, index=False)
    print(f"[INFO] Attendance saved for {subject}")

# ================= FACE RECOGNITION =================
def recognize_faces(camera_index):
    cap = cv2.VideoCapture(camera_index)
    present_names = set()

    start_time = time.time()
    frame_count = 0

    print("[INFO] Starting camera...")

    while (time.time() - start_time) < SESSION_DURATION_MIN * 60:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        results = []

        # 🔥 Run every 10 frames (performance boost)
        if frame_count % 10 == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            try:
                results = DeepFace.find(
                    img_path=small_frame,
                    db_path=IMAGES_PATH,
                    model_name="ArcFace",
                    enforce_detection=False,
                    silent=True
                )
            except:
                results = []

        if results:
            for df_res in results:
                if not df_res.empty:
                    identity = df_res.iloc[0]["identity"]
                    distance = df_res.iloc[0]["distance"]

                    x = int(df_res["source_x"][0])
                    y = int(df_res["source_y"][0])
                    w = int(df_res["source_w"][0])
                    h = int(df_res["source_h"][0])

                    if distance < 0.6:
                        # ✅ Known face
                        name = os.path.splitext(os.path.basename(identity))[0]
                        present_names.add(name)

                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                        cv2.putText(frame, name, (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0,255,0), 2)

                    else:
                        # 🔴 Unknown face
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
                        cv2.putText(frame, "UNKNOWN", (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0,0,255), 2)

                        print("⚠️ Unknown person detected")

                        # OPTIONAL: save unknown faces
                        os.makedirs("unknown", exist_ok=True)
                        cv2.imwrite(f"unknown/unknown_{int(time.time())}.jpg", frame)

        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return present_names

# ================= MAIN =================
if __name__ == "__main__":
    print("[INFO] System running...")

    while True:
        subject = get_current_class()

        if subject:
            print(f"[INFO] Starting attendance for {subject}")
            names = recognize_faces(CAMERA_INDEX)
            mark_attendance(subject, names)

            print("[INFO] Waiting for next class...")
            time.sleep(60)
        else:
            print("[INFO] No class now... waiting")
            time.sleep(IDLE_SLEEP_SEC)