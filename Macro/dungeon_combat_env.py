import time

import cv2
import gym
import numpy as np
import pyautogui
from gym import spaces
from ultralytics import YOLO

from minimap_extractor import MinimapExtractor
from skill_module import manager as skill_manager


# 아이템 드랍 보상방 뿐만 아니라 보스방에서도 나온다!! 그에 대한 보상 정책 맞는지 확인

class DungeonCombatEnv(gym.Env):
    def __init__(self, main_model_path, player_dot_model_path):
        super().__init__()
        self.yolo_model = YOLO(main_model_path)
        self.player_dot_model_path = player_dot_model_path
        self.minimap_extractor = MinimapExtractor(player_dot_model_path)

        self.observation_space = spaces.Box(low=0, high=1, shape=(20,), dtype=np.float32)
        self.action_list = [
            "Up Arrow", "Down Arrow", "Left", "Right", "Jump",
            "Flash Jump", "down_jump", "Left Flash Jump",
            "Buff Combo", "Shell Bar Combo",
            "Mid Boss Combo Shell", "Mid Boss Combo Direct",
            "Attack Normal", "Projectile Attack"
        ]
        self.action_space = spaces.Discrete(len(self.action_list))

        self.offset_x = 0
        self.offset_y = 0
        self.camera = None
        self.latest_frame = None
        self.yolo_result = None

        self.skill_log = []
        self.prev_yolo_state = {}
        self.step_count = 0
        self.portal_count = 0
        self.action_log = []
        self.hit_object_attacked = False
        self.reward_box_attacked = False
        self.dialog_reward_given = False
        self.ladder_used = False

        skill_manager.release_all(pyautogui)
        skill_manager.reset_all_timestamps()

        self.map_type = "town"
        self.fsm_state = "idle"
        self.fsm_log_path = "ppo_logs/fsm_state.log"
        self.prev_yolo_result = None
        self.prev_fsm_state = "idle"  # FSM 상태 변화 감지용
        self.fsm_entry_step = 0
        self.fsm_fail_penalty_applied = False
        self.last_skill_time = time.time()

    def set_camera(self, camera):
        self.camera = camera

    def set_offset(self, offset_xy):
        self.offset_x, self.offset_y = offset_xy

    def _refresh_latest_frame(self):
        if self.camera:
            time.sleep(0.045)  # 기존 총 sleep 시간 유지
            self.latest_frame = self.camera.get_latest_frame()
            self.yolo_result = self.yolo_model.predict(self.latest_frame, conf=0.2, imgsz=640, verbose=False)[0]
        else:
            self.latest_frame = None
            self.yolo_result = None

    def _get_player_dot_pos(self):
        minimap_img = self.minimap_extractor.get_minimap(self.latest_frame)
        bbox = self.minimap_extractor.get_player_dot_bbox(minimap_img)
        if bbox:
            x1, y1, x2, y2 = bbox
            w, h = 320, 320  # fine-tuned model 기준
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            return [cx, cy]
        return [0.0, 0.0]

    # def _get_ground_mask_penalty(self, pos):
    #     if self.latest_frame is None:
    #         return 0.0
    #
    #     minimap = self.minimap_extractor.get_minimap(self.latest_frame)
    #     if minimap is None:
    #         return 0.0
    #
    #     mask = extract_ground_mask(minimap)
    #     h, w = mask.shape[:2]
    #     px = int(pos[0] * w)
    #     py = int(pos[1] * h)
    #
    #     if 0 <= px < w and 0 <= py < h:
    #         value = mask[py, px]
    #         if value == 0:
    #             return -1.0
    #     return 0.0

    def _get_yolo_state(self):
        if self.yolo_result is None:
            return {}
        class_counts = {}
        for cls_id in self.yolo_result.boxes.cls:
            cls_name = self.yolo_model.names[int(cls_id)]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        print(f"[YOLO STATE] {class_counts}")
        return class_counts

    def _match_sequence(self, sequence):
        return any(
            self.action_log[i:i + len(sequence)] == sequence for i in range(len(self.action_log) - len(sequence) + 1))

    def _cooldown_ratio(self, skill_name, now):
        skill = skill_manager.skills[skill_name]
        elapsed = now - skill.last_used
        return np.clip(1.0 - elapsed / skill.cooldown, 0.0, 1.0)

    def _get_observation(self):
        state = self.prev_yolo_state
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        obs[0:2] = self._get_player_dot_pos()
        obs[2] = self.portal_count
        obs[3] = state.get("npc", 0)
        obs[4] = state.get("monster", 0) / 20.0
        obs[5] = int("mid_boss" in state)
        obs[6] = int("shell_bar" in state)
        obs[7] = 0
        obs[8] = int("reward_box" in state)
        obs[9] = int("hit_object" in state)
        obs[10] = state.get("item_drop", 0) / 10.0
        obs[11] = state.get("portal", 0)
        obs[12] = int(skill_manager.skills["Attack Normal"].active)
        obs[13] = int(skill_manager.skills["Attack Burst"].active)
        obs[14] = int(skill_manager.skills["Projectile Attack"].active)

        now = time.time()
        obs[15] = self._cooldown_ratio("Buff Combo", now)
        # obs[16] = self._cooldown_ratio("Install", now)
        obs[17] = 0.0
        obs[18] = 0.0
        obs[19] = float(self._is_town())

        return obs

    def _is_town(self):
        if self.latest_frame is None:
            return False
        town_tmpl = cv2.imread("template/town.png", cv2.IMREAD_GRAYSCALE)
        frame_gray = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(frame_gray, town_tmpl, cv2.TM_CCOEFF_NORMED)
        return np.max(res) > 0.8

    def _click_npc(self):
        if self.yolo_result is None:
            return False
        for box, cls in zip(self.yolo_result.boxes.xyxy, self.yolo_result.boxes.cls):
            if self.yolo_model.names[int(cls)] == "npc":
                x1, y1, x2, y2 = map(int, box[:4])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                screen_cx = self.offset_x + cx
                screen_cy = self.offset_y + cy
                print(f"[CLICK NPC] ({cx},{cy}) → 화면({screen_cx},{screen_cy})")
                pyautogui.click(screen_cx, screen_cy)
                time.sleep(0.2)  # [DELAY] 안정적인 NPC 반응 대기
                break

    def _dialog_active(self):
        if self.latest_frame is None:
            return False
        frame_gray = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2GRAY)
        templates = ["template/dialog.png", "template/dialog2.png"] if self._is_town() else ["template/dialog3.png"]
        for tmpl_path in templates:
            tmpl = cv2.imread(tmpl_path)
            tmpl = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(frame_gray, tmpl, cv2.TM_CCOEFF_NORMED)
            if np.max(res) > 0.8:
                if tmpl is None:
                    print(f"[ERROR] Template not found: {tmpl_path}")
                    continue
                h_t, w_t = tmpl.shape
                top_left = np.unravel_index(np.argmax(res), res.shape)
                cx_screen = self.offset_x + top_left[1] + w_t // 2
                cy_screen = self.offset_y + top_left[0] + h_t // 2
                pyautogui.click(cx_screen, cy_screen)
                return True
        return False

    def _handle_dialog(self):
        # ✅ NPC가 보이면 항상 먼저 클릭을 시도
        if self.prev_yolo_state.get("npc", 0) > 0:
            # 이미 대화창이 열려 있다면 클릭하지 않고 True 반환
            if self._dialog_active():
                return True
            # 대화창이 없으면 클릭 시도 → 이후 다시 체크
            self._click_npc()
            time.sleep(1.5)  # ✅ 클릭 후 프레임 안정화 대기
            return self._dialog_active()
        return False

    def _handle_action_with_guard(self, action):
        prev_state = self.prev_yolo_state.copy()

        # ✅ 대화창이 열려있으면 skill 실행 차단
        if not self._dialog_active():
            skill_manager.execute_skill(action, pyautogui, prev_state)
        else:
            print(f"[GUARD] {action} 입력 차단: 대화창 열림 상태")

        self.step_count += 1
        return prev_state

    def _update_state(self):
        curr_state = self._get_yolo_state()
        self.prev_yolo_state = curr_state.copy()

        if self._is_town():
            self.map_type = "town"
        elif self.portal_count == 6:
            self.map_type = "reward"
        else:
            self.map_type = "normal"

        return curr_state

    def _is_player_near(self, target_label, threshold=40):
        if self.yolo_result is None:
            return False
        player_center = None
        for box, cls in zip(self.yolo_result.boxes.xyxy, self.yolo_result.boxes.cls):
            name = self.yolo_model.names[int(cls)]
            x1, y1, x2, y2 = box[:4]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if name == "player":
                player_center = (cx, cy)
                break
        if player_center is None:
            return False
        for box, cls in zip(self.yolo_result.boxes.xyxy, self.yolo_result.boxes.cls):
            name = self.yolo_model.names[int(cls)]
            if name == target_label:
                x1, y1, x2, y2 = box[:4]
                tx, ty = (x1 + x2) / 2, (y1 + y2) / 2
                dist = ((player_center[0] - tx) ** 2 + (player_center[1] - ty) ** 2) ** 0.5
                if dist <= threshold:
                    return True
        return False

    def _get_distance_to(self, target_label):
        if self.yolo_result is None:
            return None

        player_center = None
        target_center = None

        for box, cls in zip(self.yolo_result.boxes.xyxy, self.yolo_result.boxes.cls):
            name = self.yolo_model.names[int(cls)]
            x1, y1, x2, y2 = map(float, box[:4])
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            if name == "player":
                player_center = (cx, cy)
            elif name == target_label:
                target_center = (cx, cy)

        if player_center and target_center:
            px, py = player_center
            tx, ty = target_center
            return ((px - tx) ** 2 + (py - ty) ** 2) ** 0.5
        return None

    def step(self, action_idx):
        # ✅ 최신 프레임 1회 갱신 후 공유 (중복 방지)
        if hasattr(self, 'camera') and self.camera is not None:
            self._refresh_latest_frame()
        else:
            print("[WARNING] Camera not set. latest_frame will be None.")
            self.latest_frame = None

        if self.yolo_result is None:
            print("[YOLO Fallback] Using previous YOLO state")
            self.yolo_result = self.prev_yolo_result
        else:
            self.prev_yolo_result = self.yolo_result

        # ✅ 대화창 처리 우선 → 대화창이 뜰 때까지 에이전트 행동 차단
        if self._handle_dialog():
            reward = 0.0
            if not self.dialog_reward_given:
                if self.portal_count == 0 and self.map_type == "normal":
                    reward = 1.0
                    self.dialog_reward_given = True
                    print("[REWARD LOG] 던전 입장 직후 NPC 대화 성공 → 보상 지급")
                elif self.portal_count == 6 and "reward_box" not in self._get_yolo_state() and self.fsm_state == "idle":
                    reward = 2.0
                    self.dialog_reward_given = True
                    print("[REWARD LOG] 보상맵 퇴장 성공 보상 지급")
                else:
                    reward = -1.0
                    self.dialog_reward_given = True
                    print("[REWARD LOG] 불필요한 NPC 대화 감지 → 패널티")
                print(f"[REWARD LOG] NPC dialog activated | Step={self.step_count} | Portal={self.portal_count}")
                return self._get_observation(), reward, False, {}

            # ✅ 대화창 열려있으면 행동 차단 → 보상 없음, 입력 없음
            print(f"[DIALOG ACTIVE] 행동 차단 | Step={self.step_count}")
            return self._get_observation(), 0.0, False, {}

        # ✅ 일반 행동 루틴
        action = self.action_list[action_idx]
        self.action_log.append(action)
        if skill_manager.has_skill(action):
            self.skill_log.append(action)

        prev_state = self._handle_action_with_guard(action)
        curr_state = self._update_state()
        # ✅ 사다리 또는 로프 사용 감지
        if self._is_player_near('ladder_rope',0) and action == "Up Arrow":
            self.ladder_used = True
            print("[LADDER USED] 사다리 또는 로프 사용 감지 → 플래그 설정")
        reward = self._calculate_reward(prev_state, curr_state, action)
        done = self.step_count >= 50

        if done:
            skill_manager.release_all(pyautogui)

        # ✅ 시각화 프레임 저장 (YOLO 박스 포함)
        if hasattr(self, "latest_frame") and self.latest_frame is not None and self.yolo_result is not None:
            import cv2, os
            os.makedirs("ppo_logs/frames", exist_ok=True)
            vis_frame = self.latest_frame.copy()
            text = f"Step: {self.step_count}, Action: {action}, Reward: {reward:.2f}"
            cv2.putText(vis_frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # ✅ YOLO 결과 시각화: _refresh_latest_frame()에서 예측된 self.yolo_result 재사용
            for box, cls_id in zip(self.yolo_result.boxes.xyxy, self.yolo_result.boxes.cls):
                x1, y1, x2, y2 = map(int, box[:4])
                cls_name = self.yolo_model.names[int(cls_id)]
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(vis_frame, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imwrite(f"ppo_logs/frames/step_{self.step_count:04d}.jpg", vis_frame)

        return self._get_observation(), reward, done, {}

    def _calculate_reward(self, prev, curr, action):
        reward = 0.0

        now = time.time()
        if skill_manager.has_skill(action):
            print(f"[TIMING] {action} at {now - self.last_skill_time:.2f}s since last skill")
            self.last_skill_time = now

        # ❌ 마을에서 포탈 사용 시 패널티
        if action == "Up Arrow" and self._is_town():
            reward -= 1.0

        # ✅ 지형 밖에 위치한 경우 페널티
        # player_pos = self._get_player_dot_pos()
        # reward += self._get_ground_mask_penalty(player_pos)

        # 마을 내부에서 NPC 탐색 보상
        if self._is_town():
            if self.prev_yolo_state.get("npc", 0) == 0 and action == "Left Flash Jump":
                reward += 0.5
                print(f"[REWARD LOG] Town movement toward NPC | Step={self.step_count} | Action={action}")
            return reward  # 마을에서는 탐색 보상만, 다른 보상 차단

        monster_diff = prev.get("monster", 0) - curr.get("monster", 0)
        portal_diff = prev.get("portal", 0) - curr.get("portal", 0)
        drop_diff = prev.get("item_drop", 0) - curr.get("item_drop", 0)

        # ✅ 던전 입장 직후 버프 사용 보상 (마을이 아닐 때만)
        if self.portal_count == 0 and not self._is_town() and "monster" in curr and action == "Buff Combo":
            reward += 1.0

        # FSM 상태 전이 감지 및 로그 출력 (공격 시퀀스 포함 여부로 조건 강화)
        if self.fsm_state == "idle" and "shell_bar" in curr and self._is_player_near(
                "shell_bar") and "Shell Bar Combo" in self.skill_log:
            self.fsm_state = "shell"
            self.fsm_entry_step = self.step_count

        elif self.fsm_state == "shell" and "shell_bar" not in curr and "mid_boss" in curr and self._is_player_near(
                "mid_boss"):
            if "Mid Boss Combo Shell" in self.skill_log:
                self.fsm_state = "mid"
                self.fsm_entry_step = self.step_count

        elif self.fsm_state == "idle" and "mid_boss" in curr and self._is_player_near(
                "mid_boss") and "Mid Boss Combo Direct" in self.skill_log:
            self.fsm_state = "mid"
            self.fsm_entry_step = self.step_count

        elif self.fsm_state == "mid" and "mid_boss" not in curr and "item_drop" in curr and self.portal_count == 5:
            if "Mid Boss Combo Direct" in self.skill_log:
                self.fsm_state = "drop"
                self.fsm_entry_step = self.step_count

        elif self.fsm_state == "mid" and "mid_boss" not in curr:
            self.fsm_state = "idle"
            self.fsm_entry_step = self.step_count

        elif self.fsm_state == "drop" and "item_drop" not in curr:
            self.fsm_state = "idle"
            self.fsm_entry_step = self.step_count
            reward += 2.5  # drop 상태 종료 보상

        # FSM 실패 탐지: 일정 시간 내 상태 전이 없을 경우 패널티 부여
        if not self.fsm_fail_penalty_applied and self.fsm_state != "idle" and self.step_count - self.fsm_entry_step > 20:
            reward -= 1.0
            print("[FSM FAIL] 상태 정체로 패널티 부여")
            self.fsm_fail_penalty_applied = True

        if self.fsm_state != self.prev_fsm_state:
            print(f"[FSM] 상태 전이: {self.prev_fsm_state} → {self.fsm_state}")
            print(f"[FSM] 직전 스킬 시퀀스: {self.action_log[-5:]}")
            if self.prev_fsm_state == "shell":
                reward += 4.0
                skill_manager.release_hold("Attack Burst", pyautogui)
            elif self.prev_fsm_state == "mid":
                reward += 2.5
                skill_manager.release_hold("Attack Normal", pyautogui)
            elif self.fsm_state == "drop":
                reward += 3.0
            self.prev_fsm_state = self.fsm_state
            self.fsm_fail_penalty_applied = False

        with open(self.fsm_log_path, "a") as f:
            f.write(
                f"Step={self.step_count}, FSM={self.fsm_state}, MapType={self.map_type}, Portal={self.portal_count}, Actions={self.action_log[-5:]}, Reward={reward:.2f}\n")

        # monster 처치시 보상
        if monster_diff > 0:
            if self.fsm_state == "shell" or self.fsm_state == "mid":
                reward += 0.1
            elif self._is_player_near("monster") and action == "Attack Normal":
                reward += 1.0

        if self.map_type == "reward" and "reward_box" in curr and self._is_player_near(
                "reward_box") and not self.reward_box_attacked:
            skill_manager.execute_skill("Projectile Attack", pyautogui, curr)
            reward += 0.5

        if self.map_type == "reward" and "reward_box" not in curr and "item_drop" in curr:
            skill_manager.release_hold("Projectile Attack", pyautogui)
            self.reward_box_attacked = True
            self.fsm_state = "drop"
            self.fsm_entry_step = self.step_count


        if self.map_type == "normal" and "hit_object" in curr and self._is_player_near(
                "hit_object") and not self.hit_object_attacked:
            skill_manager.execute_skill("Projectile Attack", pyautogui, curr)
            reward += 1.0

        if self.map_type == "normal" and "hit_object" not in curr:
            skill_manager.release_hold("Projectile Attack", pyautogui)
            self.hit_object_attacked = True
            reward += 1.0


        if self._is_player_near("item_drop") and drop_diff > 0:
            reward += 2.5

        if self._is_player_near("portal") and action == "Up Arrow" and portal_diff > 0:
            self.portal_count += 1
            reward += 1.5
            print(f"[PORTAL] Portal count increased to {self.portal_count}")

            # ✅ 하단 점프 보상 (조건 정비)
        if action == "down_jump":
            if self.portal_count == 1 and "mid_boss" not in curr:
                reward += 0.2
                print("[REWARD LOG] down_jump 시도 보상 (1번 맵)")
            elif self.portal_count == 6 and "reward_box" not in curr:
                reward += 0.2
                print("[REWARD LOG] down_jump 시도 보상 (보상맵)")
            elif self.portal_count == 1 and "mid_boss" in curr:
                reward += 1.0
                print("[REWARD LOG] down_jump: 1번 맵에서 mid_boss 등장 감지 → 보상")
            elif self.portal_count == 6 and "reward_box" in curr:
                reward += 1.0
                print("[REWARD LOG] down_jump: 보상맵에서 reward_box 등장 감지 → 보상")

        # ✅ 위치 기반 중간 보상
        if self._is_player_near("shell_bar", threshold=40):
            reward += 0.2
        if self._is_player_near("mid_boss", threshold=40):
            reward += 0.2
        if self._is_player_near("item_drop", threshold=40):
            reward += 0.3
        if self._is_player_near("reward_box", threshold=40) or self._is_player_near("hit_object", threshold=40):
            reward += 0.3
        if self._is_player_near("ladder_rope", threshold=0) and self.portal_count == 1 and not self.ladder_used:
            reward += 0.3
        if self.ladder_used:
            reward += 0.1
            dist = self._get_distance_to("portal")
            if dist is not None:
                if dist < 20:
                    reward += 1.0
                elif dist < 40:
                    reward += 0.6
                elif dist < 70:
                    reward += 0.3
                elif dist < 100:
                    reward += 0.2
                else:
                    # ✅ ladder 사용 자체에 대한 기본 보상 (100 이상인 경우)
                    reward += 0.1
            self.ladder_used = False
        if self._is_player_near("portal", threshold=0):
            reward += 0.3

        if reward > 0:
            print(
                f"[REWARD LOG] Reward={reward:.2f} | Step={self.step_count} | Action={action} | Portal={self.portal_count}")
        return reward

    def reset(self):
        if hasattr(self, 'camera') and self.camera is not None:
            self.latest_frame = self.camera.get_latest_frame()
            self.prev_yolo_state = self._get_yolo_state()
        else:
            print("[WARNING] Camera not set. latest_frame will be None.")
            self.latest_frame = None
            self.prev_yolo_state = {}
        self.step_count = 0
        self.portal_count = 0
        self.action_log = []
        self.skill_log = []
        self.hit_object_attacked = False
        self.reward_box_attacked = False
        self.dialog_reward_given = False
        self.map_type = "town"
        self.fsm_state = "idle"
        self.prev_fsm_state = "idle"
        self.fsm_entry_step = 0
        self.fsm_fail_penalty_applied = False
        self.prev_yolo_result = None
        self.last_skill_time = time.time()

        skill_manager.release_all(pyautogui)
        skill_manager.reset_all_timestamps()
        return self._get_observation()

    def seed(self, seed=None):
        import random
        import torch
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if hasattr(self.action_space, "seed"):
            self.action_space.seed(seed)
        return [seed]

    def _save_visual_step(self, action, reward):
        if self.latest_frame is not None:
            frame = self.latest_frame.copy()
            text = f"Step: {self.step_count}, Action: {action}, Reward: {reward:.2f}"
            cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imwrite(f"ppo_logs/frames/step_{self.step_count:04d}.jpg", frame)
