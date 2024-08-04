import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import matplotlib
import matplotlib as mpl
import cv2

mpl.rcParams["savefig.pad_inches"] = 0
matplotlib.rcParams["animation.embed_limit"] = 2**128

plt.ioff()


def update(frame):
    plt.clf()
    plt.imshow(frame, cmap="viridis")
    plt.axis("off")
    plt.tight_layout()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.gca().set_axis_off()


def create_video(frames):
    ratio = frames[0].shape[0] / frames[0].shape[1]
    constant = 8
    fig = plt.figure(figsize=(constant, constant * ratio))
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=60)
    html_output = HTML(ani.to_jshtml())
    plt.close(fig)
    return html_output


def save_video(frames, video_name, fps=30):
    out = cv2.VideoWriter(
        f"{video_name}.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frames[0].shape[1], frames[0].shape[0]),
    )
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()
