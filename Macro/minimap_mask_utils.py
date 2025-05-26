# minimap_mask_utils.py

import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_ground_mask(minimap_img: np.ndarray) -> np.ndarray:
    """
    minimap 이미지에서 지형(땅, 발판, 사다리 등)을 마스킹하여 반환
    흰색(255): 이동 가능한 지형
    검정(0): 배경 또는 공중
    """
    gray = cv2.cvtColor(minimap_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, thresh=80, maxval=255, type=cv2.THRESH_BINARY_INV)
    return mask

def show_mask_overlay(image: np.ndarray, mask: np.ndarray):
    """
    마스크 결과를 원본 위에 빨간색으로 시각화
    """
    overlay = image.copy()
    overlay[mask == 255] = [0, 0, 255]  # 지형 위치를 빨간색으로 표시
    blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    plt.title("Ground Mask Overlay")
    plt.axis("off")
    plt.show()

# 예시:
# img = cv2.imread("minimap_sample.png")
# mask = extract_ground_mask(img)
# show_mask_overlay(img, mask)
