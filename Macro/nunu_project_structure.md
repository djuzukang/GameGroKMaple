
### 📆 `nunu_deep_learning` 프로젝트 구조 설명

> 메이플스토리 월드 "누누월드" 강화를 위한 YOLO + 강화학습 기반 자동화 프로젝트  
> (개체 인식가 + FSM 기반 행동 제어 + 강화학습 훈련)

---

## 🧭 프로젝트 개요

- **YOLOv8**으로 인게임 객체 탐지 (player, monster, portal 등)
- **강화학습 (Stable-Baselines3 PPO)**으로 플레이어 행동 최적화
- **FSM + skill manager**로 실제 키입력 기반 제어
- **미니맵 기반 player 위치 추적 (player_dot)**

---

## 🗂️ 주요 구성 파일 설명

### 🎮 강화학습 및 환경

| 파일 | 설명 |
|------|------|
| `main_train_RI.py` | PPO 에이전트 학습 스크립트. camera / 환경 초기화 / callback 포함 |
| `dungeon_combat_env.py` | 메인 환경 클래스. YOLO 결과 기반 FSM 및 보상 계산 포함 |
| `env_wrapper.py` | TensorBoard 로깅 및 pre_step_hook 지원 래퍼 |
| `skill_module.py` | 실제 키 입력에 대응하는 스킬 정의 및 복합 스킬 루틴 관리 |

---

### 🧠 개체 인식 및 YOLO 연동

| 파일 | 설명 |
|------|------|
| `maple_env_yolo.py` | YOLO 기반 위치 encoding (player_dot 중심) |
| `minimap_extractor.py` | 미니맵 영역 자동 추출 및 player_dot 위치 예측 |
| `labeling_guidelines.md` | YOLO 클래스 정의 및 레이블링 주의사항 총정리 |

---

### 📸 데이터 수집 및 전처리

| 파일 | 설명 |
|------|------|
| `capture_images.py` | 게임 화면 캡처 및 train/val 분할 저장 |
| `dataset_structure_setup.py` | YOLO 학습을 위한 폴더 구조 및 세션 병합 관리 |

---

### 🧪 기타 유틸리티

| 파일 | 설명 |
|------|------|
| `lie_detect_solver.py` | EasyOCR 기반 거짓말 탐지 미니게임 자동화 |
| `events.out.tfevents...` | TensorBoard 학습 로그 파일 |

---

## 🔍 YOLO 클래스 목록 요약

```
npc, player, player_dot, monster, mid_boss, shell_bar,
portal, ladder_rope, hit_object, reward_box, item_drop,
skill_attack_normal, skill_attack_burst, skill_install, projectile_attack
```

> ⚠️ 현재 강화학습에서 `skill_*` 계열은 YOLO 결과를 사용하지 않음  
> `player_dot`은 관측값엔 포함되나 보상 계산에는 직접 사용되지 않음

---

## 📈 강화학습 구조 요약

- 상태 관측: YOLO 결과 + minimap 기반 player 위치 + 스킬 상태
- 행동 선택: `SkillManager` 기반 복합/단일 스킬 실행
- 보상 구조:
  - 포탈 이동
  - 중간보스 처치
  - 아이템 드랍 감지
  - reward_box 공격 성공
  - FSM 상태 전이
- 학습 방식: `Stable-Baselines3 PPO` + TensorBoard 로깅
