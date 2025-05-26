# main_train_RI.py 개선 버전
import dxcam
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from dungeon_combat_env import DungeonCombatEnv
from env_wrapper import EnvWrapper

# ✅ 모델 경로 설정
YOLO_MAIN_MODEL = "dataset/runs_up/detect/nunu_elna_exp1/weights/best.pt"
YOLO_PLAYER_DOT_MODEL = "dataset/runs_up/fine_tune/player_dot_finetune/weights/best.pt"

# ✅ 카메라 설정 (QHD 모니터 기준 좌상단 1/4)
screen_width, screen_height = 2560, 1440
region = (0, 0, screen_width // 2, screen_height // 2)
camera = dxcam.create(output_idx=0, region=region)
camera.start()

# ✅ 환경 초기화
env = DungeonCombatEnv(main_model_path=YOLO_MAIN_MODEL, player_dot_model_path=YOLO_PLAYER_DOT_MODEL)
env.set_camera(camera)
env.set_offset((region[0], region[1]))

# ✅ 래퍼 적용 (TensorBoard 포함)
env = EnvWrapper(env, log_dir="ppo_logs")


# ✅ PPO 모델 초기화
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="ppo_logs"
)

# ✅ 평가 콜백 설정
eval_callback = EvalCallback(
    env,
    best_model_save_path="ppo_log!s/best_model",
    log_path="ppo_logs/eval",
    eval_freq=1000,
    deterministic=True,
    render=False
)

# ✅ 학습 시작
model.learn(total_timesteps=100_000, callback=eval_callback)

# ✅ 저장
model.save("ppo_logs/final_model")
print("✅ PPO 학습 완료 및 저장")