import streamlit as st
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import sqlite3
import os
from datetime import datetime
import csv

# Cấu hình
DB_FILE = "faces_pro.db"
CSV_FILE = "attendance_pro.csv"
SIMILARITY_THRESHOLD = 0.38

# Khởi tạo InsightFace
@st.cache_resource
def load_insightface():
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

app = load_insightface()

# Khởi tạo database
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        code TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        department TEXT
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS face_embeddings (
        emb_id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id INTEGER,
        embedding BLOB NOT NULL,
        FOREIGN KEY(employee_id) REFERENCES employees(id)
    )
''')
conn.commit()

# Tạo CSV nếu chưa có
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Tên", "Phòng ban", "Hành động", "Thời gian", "Ngày"])

# Hàm lấy thông tin nhân viên
def get_employee_info(emp_id):
    cursor.execute("SELECT code, name, department FROM employees WHERE id = ?", (emp_id,))
    result = cursor.fetchone()
    if result:
        return {"code": result[0], "name": result[1], "department": result[2] or "Chưa có"}
    return None

# Hàm so sánh embedding
def find_best_match(emb):
    cursor.execute("SELECT employee_id, embedding FROM face_embeddings")
    stored = cursor.fetchall()

    best_sim = -1.0
    best_emp_id = None

    for emp_id, emb_blob in stored:
        stored_emb = np.frombuffer(emb_blob, dtype=np.float32)
        sim = np.dot(emb, stored_emb)
        if sim > best_sim:
            best_sim = sim
            best_emp_id = emp_id

    return best_emp_id, best_sim

# Giao diện Streamlit
st.title("Ứng dụng Chấm Công Khuôn Mặt - InsightFace")
st.write("Chụp hoặc upload ảnh để chấm công. Hỗ trợ nhiều khuôn mặt trong 1 ảnh.")

# Phần đăng ký nhân viên
st.header("1. Đăng ký nhân viên mới")
code = st.text_input("Mã nhân viên (ví dụ: NV001)")
name = st.text_input("Tên nhân viên")
department = st.text_input("Phòng ban", value="Chưa có")

if st.button("Đăng ký nhân viên"):
    if code and name:
        cursor.execute("SELECT id FROM employees WHERE code = ?", (code,))
        if cursor.fetchone():
            st.error("Mã nhân viên đã tồn tại!")
        else:
            cursor.execute("INSERT INTO employees (code, name, department) VALUES (?, ?, ?)",
                           (code, name, department))
            conn.commit()
            emp_id = cursor.lastrowid
            st.success(f"Đăng ký thành công! ID: {emp_id}")
    else:
        st.warning("Vui lòng nhập đầy đủ mã và tên.")

# Phần upload ảnh đăng ký
st.header("2. Upload ảnh đăng ký cho nhân viên")
emp_id_upload = st.number_input("ID nhân viên cần thêm ảnh", min_value=1, step=1)
uploaded_files = st.file_uploader("Chọn nhiều ảnh", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if st.button("Lưu ảnh đăng ký") and uploaded_files and emp_id_upload:
    count = 0
    for uploaded_file in uploaded_files:
        img_bytes = uploaded_file.read()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        faces = app.get(img)
        if len(faces) > 0:
            emb = faces[0].normed_embedding
            cursor.execute("INSERT INTO face_embeddings (employee_id, embedding) VALUES (?, ?)",
                           (emp_id_upload, emb.tobytes()))
            count += 1

    conn.commit()
    st.success(f"Đã lưu {count} embedding cho nhân viên ID {emp_id_upload}")

# Phần chấm công
st.header("3. Chấm công bằng khuôn mặt")
uploaded_test = st.file_uploader("Chụp hoặc upload ảnh để chấm công", type=["jpg", "jpeg", "png"])

if uploaded_test:
    img_bytes = uploaded_test.read()
    frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    faces = app.get(frame)
    if len(faces) == 0:
        st.warning("Không detect được khuôn mặt nào.")
    else:
        st.image(frame, channels="BGR", caption="Ảnh gốc")

        for face in faces:
            emb = face.normed_embedding
            bbox = face.bbox.astype(int)

            emp_id, sim = find_best_match(emb)

            if sim >= 0.38 and emp_id:
                info = get_employee_info(emp_id)
                text = f"ID: {info['code']} | {info['name']} | {info['department']}"
                color = (0, 255, 0)

                # Ghi chấm công
                now = datetime.now()
                action = "Check-in" if now.hour < 12 else "Check-out"
                t = now.strftime("%H:%M:%S")
                d = now.strftime("%Y-%m-%d")

                with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([emp_id, info['name'], info['department'], action, t, d, round(sim, 3)])

                st.success(f"Chấm công thành công: {text} - {action} ({round(sim, 3)})")
            else:
                text = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, text, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        st.image(frame, channels="BGR", caption="Kết quả nhận diện")
