import glob
import os  # Import os module to handle file deletion
import re
from PIL import Image


    # 숫자를 기준으로 정렬하는 함수
def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


for i in range(11):
    frames = []
    a = i * 50
    imgs = glob.glob(f"frame_target_{a}_*.png")  # 이미지 파일 불러오기
    imgs.sort(key=natural_keys)  # 자연스러운 숫자 순서로 정렬

    for j in imgs:
        new_frame = Image.open(j)
        frames.append(new_frame)

    frame_count = len(frames)
    # GIF 저장 (fps를 제어하려면 duration을 조정)
    gif_path = os.path.join(f'cartpole_gif_{a}_{frame_count}.gif')
    frames[0].save(gif_path, format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=50, loop=1)

    # PNG 파일 삭제
    for j in imgs:
        os.remove(j)
