from ultralytics import YOLO
import os
from pathlib import Path
import subprocess
import threading


# ✅ TensorBoard 자동 실행 (백그라운드)
def launch_tensorboard(logdir='runs/detect', port=6006):
    def _run():
        subprocess.run(["tensorboard", "--logdir", logdir, f"--port={port}"])
    threading.Thread(target=_run, daemon=True).start()
    print(f"[🔎] TensorBoard launched at http://localhost:{port}")

# ✅ 학습 실행
def train_yolo():
    model = YOLO("yolov8s.pt")  # 또는 yolov8n.pt, yolov8m.pt, yolov8l.pt 등 변경 가능
    model.train(
        data="data.yaml",         # data.yaml 경로
        epochs=50,               # 에폭 수 (로컬은 100 추천, 큰 환경에서는 증가 가능)
        imgsz=640,                # 이미지 입력 크기 (YOLO는 정사각형으로 resize)
        batch=32,                 # 로컬 GPU에 맞춘 배치 사이즈
        name="nunu_elna_exp1",    # 결과 저장 폴더 이름 (runs/detect/nunu_elna_exp1)
        project='runs/detect',    # runs/detect/exp1 구조 중 상위 폴더를 지정, 기본값은 runs/train
        # device=0,                 # 0번 GPU 사용
        # workers=8,                # 데이터 로딩 병렬 작업 수 (RAM 32GB면 여유 있음)
        verbose=True
    )


if __name__ == '__main__':
    # 🎯 로그 디렉토리 설정 및 TensorBoard 실행
    logdir = Path('runs/detect')
    if not logdir.exists():
        logdir.mkdir(parents=True, exist_ok=True)

    launch_tensorboard(logdir=str(logdir))
    train_yolo()