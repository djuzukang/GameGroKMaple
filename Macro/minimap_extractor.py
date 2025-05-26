import cv2
import numpy as np
import os
from minimap_mask_utils import extract_ground_mask, show_mask_overlay
from ultralytics import YOLO

class MinimapExtractor:
    def __init__(self, template_paths=None, tmpl_minimap_paths=None, tmpl_world_paths=None):
        self.template_paths = template_paths or ["template/bottom.png"]
        self.tmpl_minimap_paths = tmpl_minimap_paths or ["template/minimap.png", "template/minimap_2.png"]
        self.tmpl_world_paths = tmpl_world_paths or ["template/world.png", "template/world_2.png"]

        self.player_dot_model = YOLO("dataset/runs_up/fine_tune/player_dot_finetune/weights/best.pt")
        self.base_model = YOLO("dataset/runs_up/detect/nunu_elna_exp1/weights/best.pt")

    def match_template(self, image, templates):
        best_score = -1
        best_pos = None
        best_template = None
        best_w = 0
        for path in templates:
            tmpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if tmpl is None:
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > best_score:
                best_score = max_val
                best_pos = max_loc
                best_template = tmpl
                best_w = tmpl.shape[1]
        return best_pos, best_template, best_score, best_w

    def get_minimap(self, image):
        world_pos, matched_world, world_score, world_w = self.match_template(image, self.tmpl_world_paths)
        minimap_pos, matched_minimap, minimap_score, minimap_w = self.match_template(image, self.tmpl_minimap_paths)

        if not world_pos or not minimap_pos:
            return None

        wx, wy = world_pos
        mx, my = minimap_pos

        crop_x = mx
        crop_y = my + 70
        crop_w = (wx + world_w) - mx
        crop_h = 250

        img_h, img_w = image.shape[:2]
        if crop_y + crop_h > img_h:
            return None

        horizontal_slice = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        if horizontal_slice is None or horizontal_slice.size == 0:
            return None

        gray = cv2.cvtColor(horizontal_slice, cv2.COLOR_BGR2GRAY)
        line_y = self.find_horizontal_separator(gray)

        if line_y <= 0 or line_y > 250:
            return None

        minimap_img = image[crop_y:crop_y+line_y, crop_x:crop_x+crop_w]
        return minimap_img

    def find_horizontal_separator(self, gray_img: np.ndarray, search_margin: int = 60):
        height, width = gray_img.shape
        edges = cv2.Canny(gray_img, 50, 150)
        edge_sum = np.sum(edges, axis=1)
        smoothed = np.convolve(edge_sum, np.ones(5), mode='same')
        line_y = int(np.argmax(smoothed))
        return line_y

    def get_player_dot_bbox(self, minimap_crop, fallback=True) -> tuple:
        if minimap_crop is None or minimap_crop.size == 0:
            return None
        resized_crop = cv2.resize(minimap_crop, (320, 320))
        results = self.player_dot_model.predict(resized_crop, imgsz=320, conf=0.2, verbose=False)[0]
        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            if self.player_dot_model.names[int(cls)] == "player_dot":
                return tuple(map(int, box))
        if fallback:
            results = self.base_model.predict(resized_crop, imgsz=640, conf=0.2, verbose=False)[0]
            for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
                if self.base_model.names[int(cls)] == "player_dot":
                    return tuple(map(int, box))
        return None
