import time


# ✅ down_jump 복합 입력 정의
def down_jump_combo(input_controller):
    input_controller.keyDown("down")
    input_controller.press("c")
    input_controller.keyUp("down")

def left_flash_jump_combo(input_controller):
    input_controller.keyDown("left")
    input_controller.press("c")
    time.sleep(0.1)
    input_controller.press("c")
    input_controller.keyUp("left")

def mid_boss_combo_shell_combo(input_controller):
    from skill_module import manager as skill_manager
    skill_manager.skills["Weapon Swap"].execute(input_controller)
    time.sleep(0.2)
    skill_manager.skills["Install"].execute(input_controller)
    time.sleep(0.2)
    skill_manager.skills["Buff w"].execute(input_controller)
    time.sleep(0.2)
    skill_manager.skills["Attack Normal"].execute(input_controller)

def mid_boss_combo_direct_combo(input_controller):
    from skill_module import manager as skill_manager
    skill_manager.skills["Install"].execute(input_controller)
    time.sleep(0.2)
    skill_manager.skills["Buff w"].execute(input_controller)
    time.sleep(0.2)
    skill_manager.skills["Attack Normal"].execute(input_controller)

def shell_bar_combo(input_controller):
    from skill_module import manager as skill_manager
    skill_manager.skills["Weapon Swap"].execute(input_controller)
    time.sleep(0.2)
    skill_manager.skills["Attack Burst"].execute(input_controller)


def buff_combo(input_controller):
    from skill_module import manager as skill_manager
    skill_manager.skills["Buff 3"].execute(input_controller)
    time.sleep(0.1)
    skill_manager.skills["Buff d"].execute(input_controller)


class Skill:
    def __init__(self, name, key_sequence, cooldown=0, trigger=None, hold=False, complex_action=None):
        self.name = name
        self.key_sequence = key_sequence
        self.cooldown = cooldown
        self.trigger = trigger
        self.last_used = 0
        self.hold = hold
        self.active = False
        self.complex_action = complex_action

    def is_available(self, current_time, game_state):
        if current_time - self.last_used < self.cooldown:
            return False
        if self.trigger and not self.trigger(game_state):
            return False
        return True

    def execute(self, input_controller):
        if self.complex_action:
            self.complex_action(input_controller)
        for key in self.key_sequence:
            if self.hold:
                input_controller.keyDown(key)
                self.active = True
            else:
                input_controller.press(key)
        self.last_used = time.time()


# 이동기
up_arrow = Skill("Up Arrow", ["up"], cooldown=0)
down_arrow = Skill("Down Arrow", ["down"], cooldown=0.3)
right_arrow = Skill("Right", ["right"], cooldown=0)
left_arrow = Skill("Left", ["left"], cooldown=0)
jump = Skill("Jump", ["c"], cooldown=0.3)
down_jump = Skill("down_jump", [], cooldown=0.6, complex_action=down_jump_combo)
flash_jump = Skill("Flash Jump", ["c", "c"], cooldown=0.6)
left_flash_jump = Skill("Left Flash Jump", [], cooldown=0.6, complex_action=left_flash_jump_combo)

# 버프 스킬
buff_3 = Skill("Buff 3", ["3"], cooldown=1.0) #(agent는 호출 안 함)
buff_d = Skill("Buff d", ["d"], cooldown=1.0) #(agent는 호출 안 함)
buff_w = Skill("Buff w", ["w"], cooldown=1.0) #(agent는 호출 안 함)

#설치기
install = Skill("Install", ["4"], cooldown=1.0) #(agent는 호출 안 함)
#무기 스왑
weapon_swap = Skill("Weapon Swap", ["1"], cooldown=0.5) #(agent는 호출 안 함)

# 복합 스킬
buff_combo = Skill("Buff Combo", [], cooldown=60, complex_action=buff_combo)

# 공격 스킬 (키다운 유지)
skill_attack_normal = Skill("Attack Normal", ["shift"], cooldown=0.1, hold=True)
projectile_attack = Skill("Projectile Attack", ["ctrl"], cooldown=0.1, hold=True)
attack_burst = Skill("Attack Burst", ["r"], cooldown=0.1, hold=True) #(agent는 호출 안 함)

# 복합 공격 루틴
mid_boss_combo_shell = Skill("Mid Boss Combo Shell", [], cooldown=3.0, complex_action=mid_boss_combo_shell_combo)
mid_boss_combo_direct = Skill("Mid Boss Combo Direct", [], cooldown=3.0, complex_action=mid_boss_combo_direct_combo)
shell_bar_combo = Skill("Shell Bar Combo", [], cooldown=2.0, complex_action=shell_bar_combo)

class SkillManager:
    def __init__(self):
        self.skills = {}

    def add_skill(self, skill):
        self.skills[skill.name] = skill

    def execute_skill(self, skill_name, input_controller, game_state):
        skill = self.skills.get(skill_name)
        if skill and skill.is_available(time.time(), game_state):
            print(f"[SkillManager] Executing: {skill.name}")
            skill.execute(input_controller)

    def release_hold(self, skill_name, input_controller):
        skill = self.skills.get(skill_name)
        if skill and skill.hold and skill.active:
            input_controller.keyUp(skill.key_sequence[0])
            skill.active = False

    def release_all(self, input_controller):
        for skill in self.skills.values():
            if skill.hold and skill.active:
                input_controller.keyUp(skill.key_sequence[0])
                skill.active = False

    def has_skill(self, skill_name):
        return skill_name in self.skills

    def reset_all_timestamps(self):
        for skill in self.skills.values():
            skill.last_used = 0



manager = SkillManager()
for s in [
    up_arrow, down_arrow, right_arrow, left_arrow, jump, flash_jump,
    down_jump, left_flash_jump,
    buff_combo, buff_d, buff_3, buff_w, install, weapon_swap,
    skill_attack_normal, projectile_attack, attack_burst,
    mid_boss_combo_shell, mid_boss_combo_direct, shell_bar_combo
]:
    manager.add_skill(s)
