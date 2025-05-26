# dataset_structure_setup.py (최신 버전)

import os
import shutil
from tqdm import tqdm


def setup_project_structure(base_dir="dataset"):
    print(f"[~] 프로젝트 폴더 구조 생성 중... ({base_dir})")
    for dungeon in ["elna", "leafre"]:
        for split in ["train", "val"]:
            os.makedirs(os.path.join(base_dir, dungeon, "images", split), exist_ok=True)
            os.makedirs(os.path.join(base_dir, dungeon, "labels", split), exist_ok=True)
    print("[✔] 기본 폴더 구조 생성 완료!")


def select_dungeon():
    print("\n=== 던전 선택 ===")
    print("1: 엘나스 던전 (elna)")
    print("2: 리프레 던전 (leafre)")
    choice = input("던전 번호를 선택하세요: ").strip()
    return "elna" if choice == "1" else "leafre"


def merge_sessions(source_root, target_root, dungeon_type="elna"):
    print(f"[~] {source_root} 안의 세션들을 {target_root}/{dungeon_type}/images/train, val로 병합 중...")

    temp_capture_root = os.path.join(target_root, dungeon_type, "images", "temp_capture")
    final_train_root = os.path.join(target_root, dungeon_type, "images", "train")
    final_val_root = os.path.join(target_root, dungeon_type, "images", "val")

    os.makedirs(final_train_root, exist_ok=True)
    os.makedirs(final_val_root, exist_ok=True)

    if not os.path.exists(temp_capture_root):
        print(f"[!] {temp_capture_root}가 존재하지 않습니다. 먼저 캡쳐를 진행하세요.")
        return

    # 세션 폴더 순회
    for session_folder in os.listdir(temp_capture_root):
        session_path = os.path.join(temp_capture_root, session_folder)
        if not os.path.isdir(session_path):
            continue

        # train/val 각각 복사
        for split in ["train", "val"]:
            split_src = os.path.join(session_path, split)
            split_dst = final_train_root if split == "train" else final_val_root

            if os.path.exists(split_src):
                for filename in tqdm(os.listdir(split_src), desc=f"{session_folder} -> {split}"):
                    src_path = os.path.join(split_src, filename)
                    dst_path = os.path.join(split_dst, filename)
                    shutil.copy2(src_path, dst_path)  # 기존 파일 있으면 덮어쓰기

    # ✅ 병합 후 temp_capture 삭제
    shutil.rmtree(temp_capture_root)
    print(f"[✔] 병합 및 temp_capture 정리 완료!")


if __name__ == "__main__":
    setup_project_structure(base_dir="dataset")
    dungeon = select_dungeon()
    merge_sessions("dataset", "dataset", dungeon_type=dungeon)
