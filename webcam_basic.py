"""
웹캠 기본 구동 코드
사용법:
    conda activate webcam_env
    python webcam_basic.py
조작키:
    q         - 종료
    s         - 현재 화면 캡처 저장
    r         - 녹화 시작/정지
    g         - 그레이스케일 토글
    f         - 얼굴 인식 토글
"""

import cv2
import os
from datetime import datetime

# ── 설정 ──────────────────────────────────────────────
CAMERA_INDEX   = 0          # 웹캠 인덱스 (여러 대면 1, 2… 시도)
FRAME_WIDTH    = 1280
FRAME_HEIGHT   = 720
FPS            = 30
SAVE_DIR       = "captures" # 캡처/녹화 저장 폴더
# ──────────────────────────────────────────────────────

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_face_cascade() -> cv2.CascadeClassifier:
    """Haar Cascade 얼굴 인식기 로드"""
    # cv2.data 는 Pylance 스텁에 없으므로 getattr 로 접근해 경고를 억제합니다.
    haarcascades_dir: str = getattr(cv2, "data").haarcascades
    cascade_path = os.path.join(haarcascades_dir, "haarcascade_frontalface_default.xml")
    return cv2.CascadeClassifier(cascade_path)

def detect_faces(frame, cascade, gray):
    """얼굴 감지 후 사각형 표시"""
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Face", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame, len(faces)

def draw_hud(frame, fps_val, grayscale, recording, face_mode, face_count):
    """화면 상단에 상태 정보 표시 (HUD)"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    info = (
        f"FPS:{fps_val:5.1f}  "
        f"{'[GRAY] ' if grayscale else ''}"
        f"{'[REC●] ' if recording else ''}"
        f"{'[FACE:' + str(face_count) + '] ' if face_mode else ''}"
        f"| q:종료  s:캡처  r:녹화  g:흑백  f:얼굴"
    )
    cv2.putText(frame, info, (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame

def main():
    ensure_dir(SAVE_DIR)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] 카메라 인덱스 {CAMERA_INDEX} 를 열 수 없습니다.")
        print("        다른 인덱스(1, 2…)를 CAMERA_INDEX 에 지정해 보세요.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          FPS)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] 해상도: {actual_w}x{actual_h}")

    face_cascade = get_face_cascade()

    # 상태 플래그
    grayscale    = False
    face_mode    = False
    recording    = False
    writer: cv2.VideoWriter | None = None
    face_count   = 0

    # FPS 측정용
    tick       = cv2.getTickCount()
    fps_val    = 0.0
    frame_cnt  = 0

    print("[INFO] 웹캠 시작. 'q' 키로 종료합니다.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 프레임을 읽지 못했습니다.")
            break

        # FPS 계산 (30프레임마다 갱신)
        frame_cnt += 1
        if frame_cnt % 30 == 0:
            now      = cv2.getTickCount()
            fps_val  = 30 * cv2.getTickFrequency() / (now - tick)
            tick     = now

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 얼굴 인식
        if face_mode:
            frame, face_count = detect_faces(frame, face_cascade, gray)

        # 그레이스케일 출력
        display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if grayscale else frame

        # HUD 오버레이
        display = draw_hud(display, fps_val, grayscale, recording, face_mode, face_count)

        # 녹화 중이면 원본(컬러) 프레임 기록
        if recording and writer is not None:
            writer.write(frame)

        cv2.imshow("Webcam  |  q:종료  s:캡처  r:녹화  g:흑백  f:얼굴", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("[INFO] 종료합니다.")
            break

        elif key == ord('s'):
            path = os.path.join(SAVE_DIR, f"capture_{timestamp()}.jpg")
            cv2.imwrite(path, frame)
            print(f"[INFO] 캡처 저장: {path}")

        elif key == ord('r'):
            if not recording:
                path    = os.path.join(SAVE_DIR, f"video_{timestamp()}.mp4")
                fourcc  = int(cv2.VideoWriter_fourcc(*"mp4v"))  # type: ignore[attr-defined]
                writer  = cv2.VideoWriter(path, fourcc, FPS, (actual_w, actual_h))
                recording = True
                print(f"[INFO] 녹화 시작: {path}")
            else:
                recording = False
                if writer is not None:
                    writer.release()
                writer = None
                print("[INFO] 녹화 종료.")

        elif key == ord('g'):
            grayscale = not grayscale
            print(f"[INFO] 그레이스케일: {'ON' if grayscale else 'OFF'}")

        elif key == ord('f'):
            face_mode = not face_mode
            print(f"[INFO] 얼굴 인식: {'ON' if face_mode else 'OFF'}")

    # 정리
    if recording and writer:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 리소스 해제 완료.")

if __name__ == "__main__":
    main()