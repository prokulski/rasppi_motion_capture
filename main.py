# %%
import argparse
import time
from collections import deque

import cv2
import numpy as np

# %%
MAX_WIDTH = 1280
MAX_HEIGHT = 720
SAVE_EVERY_N_SECONDS = 900
MOVIE_PATH_EXTENSION = ".mp4"
FOURCC = "mp4v"

# %%
parser = argparse.ArgumentParser()
parser.add_argument(
    "--threshold", "-t", type=int, default=30, help="Poziom zmiany białe-czarne"
)
parser.add_argument(
    "--diff", "-d", type=float, default=0.075, help="Minimalna różnica w %% punktów"
)
parser.add_argument("--scale", "-s", type=int, default=1, help="Krotność pomniejszenia")
parser.add_argument(
    "--moviepath",
    "-mp",
    type=str,
    default="video_output",
    help="Nazwa pliku do którego zostaną zapisane wyjściowe pliki wideo. Bez rozszerzenia!",
)


# %%
def prepare_video_out(movie_path, fourcc):
    output_movie_path = (
        f"{movie_path}_{time.strftime('%Y_%m_%d_%H_%M')}{MOVIE_PATH_EXTENSION}"
    )
    video_out = cv2.VideoWriter(output_movie_path, fourcc, 25, (height, width))
    return video_out


# %%
if __name__ == "__main__":
    args = parser.parse_args()

    print(f"{__file__}:")
    for k, v in args._get_kwargs():
        print(f"\t{k} = {v}")

    cap = cv2.VideoCapture(0)
    # dajemy czas na obudzenie się strumienia
    time.sleep(1)

    # max rozdzielczość kamery
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, MAX_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, MAX_HEIGHT)
    # autoexposure
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

    print(f"cap width = {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"cap height = {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

    # wczytanie pierwszej klatki - potrzebne są rozmiary itp
    ret, frame = cap.read()
    frame = cv2.resize(
        frame, (frame.shape[0] // args.scale, frame.shape[1] // args.scale)
    )

    height, width, channels = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*FOURCC)
    video_out = prepare_video_out(args.moviepath, fourcc)

    last_frame = None
    last_saved = time.time()
    last_p = 0
    fps_list = deque([1] * 25, maxlen=25)

    while True:
        start_time = time.time()
        ret, frame = cap.read()

        if frame is not None:
            if args.scale != 1:
                frame = cv2.resize(frame, (height, width))
            frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_bw = cv2.GaussianBlur(frame_bw, (5, 5), 0)
            _, frame_bw = cv2.threshold(
                frame_bw, args.threshold, 255, cv2.THRESH_BINARY
            )

            if last_frame is None:
                last_frame = frame_bw.copy()

            diff_np = np.int32(last_frame) - np.int32(frame_bw)
            _, c = np.unique(np.abs(diff_np), return_counts=True)
            p = 100 * c[0] / (1 + sum(c))
            delta_p = abs(last_p - p)
            last_p = p

            params = f"diff: {delta_p:.4f} | fps: {1/np.mean(fps_list):.2f}"

            if delta_p > args.diff:
                # dodajemy godzinę do obrazka
                time_part = str(time.strftime("%Y-%m-%d %H:%M:%S"))
                cv2.putText(
                    frame, time_part, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255)
                )
                cv2.putText(
                    frame, params, (10, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255)
                )
                video_out.write(frame)

            print(params, end="\r")

        # time.sleep(0.1)
        fps_list.append(time.time() - start_time)
        if last_saved < (time.time() - SAVE_EVERY_N_SECONDS):
            video_out.release()
            video_out = prepare_video_out(args.moviepath, fourcc)
            last_saved = time.time()

    cap.release()
    video_out.release()
