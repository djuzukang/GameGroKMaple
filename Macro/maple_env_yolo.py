import gym
import numpy as np
import cv2
from gym import spaces
from ultralytics import YOLO
from skill_module import SkillManager, manager as skill_manager
import pyautogui

class YoloStateEncoder:
    def __init__(self, yolo_model_path: str):
        self.model = YOLO(yolo_model_path)

    def encode_state(self, image: np.ndarray, player_class='player_dot') -> np.ndarray:
        resized_image = cv2.resize(image, (320, 320))
        results = self.model.predict(resized_image, conf=0.2, verbose=False)[0]

        player_pos = None
        for box, cls_id in zip(results.boxes.xyxy, results.boxes.cls):
            cls_name = self.model.names[int(cls_id)]
            if cls_name == player_class:
                x1, y1, x2, y2 = map(int, box[:4])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                player_pos = (cx, cy)
                break

        if player_pos is None:
            player_pos = (160, 220)

        return np.array(player_pos, dtype=np.int16)

class MapleEnvYolo(gym.Env):
    def __init__(self, model_path, target_pos=(180, 180)):
        super(MapleEnvYolo, self).__init__()

        self.yolo_encoder = YoloStateEncoder(model_path)
        self.action_list = list(skill_manager.skills.keys())
        self.action_space = spaces.Discrete(len(self.action_list))
        self.observation_space = spaces.Box(low=0, high=320, shape=(2,), dtype=np.int16)

        self.player_pos = np.array([60, 200])
        self.target_pos = np.array(target_pos)
        self.step_count = 0
        self.latest_minimap = None

    def update_frame(self, minimap_img):
        self.latest_minimap = minimap_img

    def step(self, action_idx):
        action = self.action_list[action_idx]
        prev_pos = self.player_pos.copy()

        skill_manager.execute_skill(action, pyautogui, {})
        self.step_count += 1

        if self.latest_minimap is not None:
            self.player_pos = self.yolo_encoder.encode_state(self.latest_minimap)

        dist_before = np.linalg.norm(self.target_pos - prev_pos)
        dist_after = np.linalg.norm(self.target_pos - self.player_pos)
        reward = dist_before - dist_after

        done = dist_after < 8 or self.step_count >= 50

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        if self.latest_minimap is None:
            return np.zeros(self.observation_space.shape, dtype=np.int16)
        return self.yolo_encoder.encode_state(self.latest_minimap)

    def reset(self):
        self.player_pos = np.array([60, 200])
        self.target_pos = np.array([180, 180])
        self.step_count = 0
        return self._get_obs()
