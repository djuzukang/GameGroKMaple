import dxcam
import time
import os
import shutil
import random
from datetime import datetime
from PIL import Image
import win32gui
import win32con
import win32api

# 프로젝트 폴더 설정
PROJECT_ROOT = "dataset"

# 창 활성화 함수
def activate_window(window_name):
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd == 0:
        raise Exception(f"[!] 게임 창을 찾을 수 없습니다: {window_name}")

    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width = right - left
    height = bottom - top
    print(f"[~] 현재 창 위치 및 크기: ({left}, {top}, {width}, {height})")

    if win32gui.IsIconic(hwnd):
        win32gui.ShowWindow(hwnd, win32con.SW_SHOWNOACTIVATE)
        time.sleep(0.5)

    try:
        win32gui.SetForegroundWindow(hwnd)
    except:
        print("[!] SetForegroundWindow 실패. 창 복원 시도 중...")
        win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
        time.sleep(0.5)

    print("[✔] 게임 창 활성화 완료")

# 화면 캡처 영역 설정 (좌측 상단 1/4)
def get_top_left_quarter():
    screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
    screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
    return (0, 0, screen_width // 2, screen_height // 2)

# 이미지 저장
def save_image(frame, save_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = os.path.join(save_dir, f"capture_{timestamp}.png")
    Image.fromarray(frame).save(filename)
    print(f"[+] 이미지 저장: {filename}")

# train/val 데이터 분리
def split_train_val(base_dir, train_ratio=0.8):
    images = [f for f in os.listdir(base_dir) if f.endswith(".png")]
    if not images:
        print("[!] 분리할 이미지가 없습니다.")
        return

    random.shuffle(images)
    train_count = int(len(images) * train_ratio)
    train_images = images[:train_count]
    val_images = images[train_count:]

    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for img in train_images:
        shutil.move(os.path.join(base_dir, img), os.path.join(train_dir, img))
    for img in val_images:
        shutil.move(os.path.join(base_dir, img), os.path.join(val_dir, img))

    print(f"[✔] 데이터 분리 완료 (Train: {len(train_images)}, Val: {len(val_images)})")

# 이미지 캡처 함수
def capture_images(duration, interval, save_dir, window_name):
    activate_window(window_name)
    region = get_top_left_quarter()
    print(f"[~] {duration}초 동안 {region} 영역을 {interval}초 간격으로 캡쳐합니다...")

    camera = dxcam.create(region=region)
    camera.start(target_fps=1)

    start_time = time.time()
    os.makedirs(save_dir, exist_ok=True)

    while time.time() - start_time < duration:
        frame = camera.get_latest_frame()
        if frame is not None:
            save_image(frame, save_dir)
        time.sleep(interval)

    camera.stop()
    print("[✔] 캡쳐 완료!")

# 던전 선택 함수
def select_dungeon():
    print("\n=== 던전 선택 ===")
    print("1: 엘나스 던전 (elna)")
    print("2: 리프레 던전 (leafre)")
    choice = input("던전 번호를 선택하세요: ").strip()

    if choice == "1":
        return "elna"
    elif choice == "2":
        return "leafre"
    else:
        print("[!] 잘못된 입력입니다. 기본값 'elna'로 설정합니다.")
        return "elna"

# 메인 실행
def main():
    print("=== 이미지 캡쳐 설정 ===")

    try:
        dungeon = select_dungeon()
        duration = int(input("캡쳐할 총 시간 (초, 기본 120): ") or 120)
        interval = float(input("캡쳐 간격 (초, 기본 3): ") or 3)
        window_name = input("게임 창 이름 (기본: MapleStory Worlds-메이플 누누 월드): ") or "MapleStory Worlds-메이플 누누 월드"
    except Exception as e:
        print(f"[!] 입력 오류: {e}")
        return

    timestamp_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = os.path.join(PROJECT_ROOT, dungeon, "images", "temp_capture", timestamp_folder)

    capture_images(duration, interval, save_folder, window_name)
    split_train_val(save_folder)

if __name__ == "__main__":
    main()
