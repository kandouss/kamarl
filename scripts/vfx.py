import os
import glob
import argparse
import numpy as np

import moviepy as mpy
from moviepy.editor import *
from moviepy.video.tools.segmenting import findObjects


parser = argparse.ArgumentParser(description='Stitch together some video clips.')
parser.add_argument('--train_steps','-s', metavar='train_steps', type=int, nargs='+',
                   help='training steps to include in file')
parser.add_argument('--base_dir','-d', metavar='episode_directory', type=str,required=False, default='.',
                    help='directory of training run')
parser.add_argument('--title', '-t', metavar='clip_title', type=str, required=False, default='',
                    help='Episode title')
parser.add_argument('--delay_s', metavar='delay_s', type=float, required=False, default=1,
                    help='Text delay (seconds)')
parser.add_argument('--outsize', metavar='output_size', type=int, nargs=2, required=False, default=(720, 480))
parser.add_argument('--output_file', '-o', metavar='output_file', type=str, required=False, default='out.mp4')
parser.add_argument('--fps', metavar='fps', type=int, required=False, default=20)
parser.add_argument('--time_stretch', metavar='time_stretch', type=float, required=False, default=0.5)
parser.add_argument('--target_duration', metavar='target_duration', type=float, required=False, default=5)
parser.add_argument('--captions', metavar='captions', type=str, nargs='*', required=False, default=[])
parser.add_argument('--end_delay', metavar='end_delay', type=float, required=False, default=0)


args = parser.parse_args()

base_dir = args.base_dir
train_steps = args.train_steps
title = args.title
delay_s = args.delay_s
outsize = args.outsize
output_file = args.output_file
fps = args.fps
time_stretch = args.time_stretch
target_duration = args.target_duration
captions = args.captions
end_delay = args.end_delay

if len(captions) == 0:
    captions = [""]*len(train_steps)

if len(captions) != len(train_steps):
    raise ValueError("Must provide the same number of captions as train steps.")

print("Base dir is",base_dir)

TEXT_COLOR='#bfbeba'
BOTTOM_INFO_HEIGHT=100
BOTTOM_TEXT_PADDING=20

if not os.path.isdir(base_dir):
    raise ValueError(f"Couldn't access episode directory \"{base_dir}\".")

files = []
for k in train_steps:
    tmp = os.path.join(base_dir, f'episode_{k}.mp4')
    if not os.path.isfile(tmp):
        raise ValueError(f"Couldn't access \"episode_{k}.mp4\" in directory:\n\t\"{base_dir}\".")
    files.append(tmp)


clips = []
if title:
    clips.append(
        CompositeVideoClip(
            [TextClip(title, color=TEXT_COLOR,fontsize=80,font='prollynomatch').set_position('center').subclip(0,delay_s)],
            size=outsize
        )
    )

frame_size = None
resize_width = outsize[0]
for ep_no, fpath, caption in zip(train_steps, files, captions):
    clips.append(
        CompositeVideoClip(
            [TextClip(f'Episode {ep_no}\n{caption}', color=TEXT_COLOR,fontsize=80,font='prollynomatch').set_position('center').subclip(0,delay_s),],
            size=outsize
        )
    )
    tmp = os.path.join(base_dir, f"episode_{ep_no}","frame_*.png")
    image_files = list(sorted(glob.glob(tmp), key=lambda x: int(x.split('frame_')[1].split('.png')[0])))
    episode_steps = len(image_files)-1
    fps_duration = episode_steps/fps
    calculated_duration = (1.-time_stretch)*fps_duration + (time_stretch)*target_duration
    calc_fps = episode_steps/calculated_duration
    image_files += [image_files[-1]]*(int(calc_fps*end_delay))
    clip = ImageSequenceClip(image_files, fps=calc_fps)
    print(f"Episode {ep_no}: calculated duration {calculated_duration}")

    if frame_size is None:
        frame_size = clip.get_frame(0).shape[:2][::-1]
        scaled_frame_height = (outsize[0])*(frame_size[1]/frame_size[0]) 
        if scaled_frame_height+BOTTOM_INFO_HEIGHT < outsize[1]:
            target_height = scaled_frame_height
            resize_width = outsize[0]
        else:
            target_height = outsize[1] - BOTTOM_INFO_HEIGHT
            resize_width = int(target_height * frame_size[0]/frame_size[1])

    infotext = [f'{episode_steps} steps ({int(calc_fps)} fps)', caption]
        
    textloc = (BOTTOM_TEXT_PADDING, outsize[1]-BOTTOM_INFO_HEIGHT+BOTTOM_TEXT_PADDING)
    infoloc = (outsize[0]//2+BOTTOM_TEXT_PADDING, outsize[1]-BOTTOM_INFO_HEIGHT+BOTTOM_TEXT_PADDING//2)

    clips.append(

        CompositeVideoClip(
            [
                clip.resize(width=resize_width).set_position(('center','top')),
                TextClip(f'Episode {ep_no}', color=TEXT_COLOR,fontsize=40,font='prollynomatch').set_position(textloc).subclip(0, clip.duration),
                TextClip('\n'.join(infotext), color=TEXT_COLOR,fontsize=30,font='prollynomatch').set_position(infoloc).subclip(0, clip.duration)
            ],
        size=outsize
        )
        # CompositeVideoClip(
        #     [VideoFileClip(fpath).resize(width=outsize[0])],
        #     size=outsize
        # )
    )

clips.append(
    CompositeVideoClip(
        [TextClip(f'end', color=TEXT_COLOR,fontsize=20,font='prollynomatch').set_position(('left','bottom')).subclip(0,delay_s),],
        size=outsize
    )
)

final_clip = concatenate_videoclips(clips)
final_clip.write_videofile(output_file, fps=20)