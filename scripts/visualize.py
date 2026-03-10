import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from moviepy.editor import VideoFileClip
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", default="output", help="directory containing output_*.csv files")
parser.add_argument("--out",       default=".",      help="directory to write mp4/gif files")
args = parser.parse_args()

files = sorted(glob.glob(os.path.join(args.input_dir, "output_*.csv")),
               key=lambda x: int(x.split('_')[-1].split('.')[0]))
frames = [pd.read_csv(file) for file in files]
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
fig2 = plt.figure(figsize=(6,6), facecolor='black')
ax2 = fig2.add_subplot(111, projection='3d')
ax2.set_facecolor('black')
pink_shades = plt.cm.pink(np.linspace(0.3, 0.8, len(frames[0])))

def init():
  sc = ax.scatter([], [], [], s=2)
  ax.set_xlim(0,1)
  ax.set_ylim(0,1)
  ax.set_zlim(0,1)
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  return sc,

def initPretty():
  dummy = np.zeros(len(frames))
  sc = ax2.scatter(dummy, dummy, dummy,
                    c=plt.cm.RdPu(np.linspace(0.55, 0.9, len(frames))),
                    s=20, depthshade=True)
  ax2.grid(False)
  ax2.set_xticks([])
  ax2.set_yticks([])
  ax2.set_zticks([])
  ax2.set_xlabel('')
  ax2.set_ylabel('')
  ax2.set_zlabel('')
  ax2.set_xlim(-30, 30)
  ax2.set_ylim(-30, 30)
  ax2.set_zlim(-30, 30)
  for axis in [ax2.xaxis, ax2.yaxis, ax2.zaxis]:
    axis.pane.fill = False        # remove the translucent plane faces
    axis.pane.set_edgecolor('none')  # remove edges between planes
    axis.line.set_color((1.0, 1.0, 1.0, 0.0))  # hide the axis lines
  ax2.set_proj_type('persp')  # smoother depth feel
  ax2.grid(False)
  ax2.set_box_aspect([1, 1, 1])
  return []

def update(frame_idx):
  data = frames[frame_idx]
  ax.clear() # clears previous points
  ax.set_xlim(-50, 50)
  ax.set_ylim(-50, 50)
  ax.set_zlim(-50, 50)
  ax.scatter(data['x'], data['y'], data['z'], s=5)
  return ax,

def updatePretty(frame_idx):
  for artist in ax2.collections:
      artist.remove()
  data = frames[frame_idx]
  ax2.scatter(data['x'], data['y'], data['z'], c=pink_shades,
                s=20, depthshade=False)
  return []

animation = FuncAnimation(fig, update, frames=len(frames), init_func=init,
                          interval=50, blit=False, repeat=False)
animation.save(os.path.join(args.out, 'nbody_sim.mp4'), writer='ffmpeg', fps=30)
print("saved animation as nbody_sim.mp4")

animation = FuncAnimation(fig2, updatePretty, frames=len(frames), init_func=initPretty,
                          interval=50, blit=False, repeat=False)
animation.save(os.path.join(args.out, 'nbody_pretty_sim.mp4'), writer='ffmpeg', fps=30)
print("saved animation as nbody_pretty_sim.mp4")

clip = VideoFileClip(os.path.join(args.out, "nbody_sim.mp4"))
clip.write_gif(os.path.join(args.out, "nbody_sim.gif"), fps=clip.fps)
print("saved gif as nbody_sim.gif")

clip = VideoFileClip(os.path.join(args.out, "nbody_pretty_sim.mp4"))
clip.write_gif(os.path.join(args.out, "nbody_pretty_sim.gif"), fps=clip.fps)
print("saved gif as nbody_pretty_sim.gif")
