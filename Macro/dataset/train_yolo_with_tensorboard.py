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


# ✅ fine-tuning 실행 함수 (player_dot 전용)
def fine_tune_yolo():
    model = YOLO("yolov8s.pt")  # 또는 yolov8n.pt, yolov8m.pt, yolov8l.pt 등 변경 가능
    model.train(
        data="player_dot_finetune.yaml",
        epochs=50,
        imgsz=320,
        batch=32,
        name="player_dot_finetune",
        project="runs/fine_tune",
        verbose=True
    )


if __name__ == '__main__':
    # 🎯 로그 디렉토리 설정 및 TensorBoard 실행
    logdir = Path('runs/detect')
    if not logdir.exists():
        logdir.mkdir(parents=True, exist_ok=True)

    launch_tensorboard(logdir=str(logdir))
    # train_yolo()
    fine_tune_yolo()

    # ⚠️ 강화학습용 환경에서 YOLO는 객체 인식만 수행하고
    # 이후 행동(키입력, 클릭 등)은 DQN 모델이 결정합니다.
    # 아래 추론 함수들은 실사용 루틴에는 포함하지 않아도 됩니다.

    # ✅ 병합 추론 (단일 이미지)
    def unified_predict_pipeline(original_weights="runs/detect/nunu_elna_exp1/weights/best.pt",
                                 finetuned_weights="runs/fine_tune/player_dot_finetune/weights/best.pt",
                                 source="sample.jpg",
                                 conf_orig=0.18, conf_fine=0.16):
        yolo_orig = YOLO(original_weights)
        yolo_fine = YOLO(finetuned_weights)

        result_orig = yolo_orig.predict(source=source, conf=conf_orig, save=False)[0]
        result_fine = yolo_fine.predict(source=source, conf=conf_fine, save=False)[0]

        merged_boxes = result_orig.boxes.data.tolist() + result_fine.boxes.data.tolist()
        print(f"[🔀] 객체 총 {len(merged_boxes)}개 병합됨 (orig: {len(result_orig.boxes)}, fine: {len(result_fine.boxes)})")
        return merged_boxes


    # ✅ 병합 다중 이미지 추론
    def unified_multiple_image_predict(
            folder="predict_images",
            original_weights="runs/detect/nunu_elna_exp1/weights/best.pt",
            finetuned_weights="runs/fine_tune/player_dot_finetune/weights/best.pt",
            conf_orig=0.18,
            conf_fine=0.16,
            tag="merged"
    ):
        yolo_orig = YOLO(original_weights)
        yolo_fine = YOLO(finetuned_weights)
        image_list = glob.glob(f"{folder}/*.jpg") + glob.glob(f"{folder}/*.png")

        for img_path in image_list:
            result_orig = yolo_orig.predict(source=img_path, conf=conf_orig, save=False)[0]
            result_fine = yolo_fine.predict(source=img_path, conf=conf_fine, save=False)[0]

            merged_boxes = result_orig.boxes.data.tolist() + result_fine.boxes.data.tolist()
            print(f"[🔀] {img_path} 객체 총 {len(merged_boxes)}개 병합됨")


    def real_time_capture_predict(weights_path="runs/detect/nunu_elna_exp1/weights/best.pt",
                                  conf_thres=0.18):
        model = YOLO(weights_path)
        camera = dxcam.create(output_color="BGR")
        camera.start(target_fps=10)

        region = (0, 0, 1280, 720)  # QHD 좌측 상단 1/4

        while True:
            frame = camera.get_latest_frame(region=region)
            if frame is None:
                continue

            results = model.predict(
                source=frame,
                conf=conf_thres,
                save_crop=True,
                save=False,
                show=False
            )

            for box in results[0].boxes.data:
                cls = int(box[5])
                if model.names[cls] == "projectile_attack":
                    print("[🎯] 공격 감지됨!")

            time.sleep(0.5)
