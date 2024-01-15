"""VideoDataset class is based on: https://github.com/antoine77340/MIL-NCE_HowTo100M"""

import torch as th
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import random
import ffmpeg
import subprocess


class VideoDataset(Dataset):
    """HowTo100M Video-Text loader."""

    def __init__(
            self,
            video_list,
            videos_path,
            fps,
            num_frames,
            frame_size,
            crop_only,
            center_crop,
            benchmark=False,
            random_left_right_flip=False,
    ):
        """
        Args:
        """
        assert isinstance(frame_size, int)
        self.video_list = pd.read_csv(video_list)
        self.videos_path = videos_path
        self.size = frame_size
        self.num_frames = num_frames
        self.fps = fps
        self.num_sec = self.num_frames / float(self.fps)
        self.crop_only = crop_only
        self.center_crop = center_crop
        self.benchmark = benchmark
        self.random_flip = random_left_right_flip

    def __len__(self):
        return len(self.video_list)

    def _get_video(self, video_path, video_duration):
        num_clip = int(video_duration / self.num_sec)
        video = th.zeros(num_clip, 3, self.num_frames, self.size, self.size)
        start_ind = np.linspace(0, max(0, video_duration - self.num_sec - 0.4), num_clip)
        for i, s in enumerate(start_ind):
            video[i] = self._get_video_start(video_path, s)
        return video

    def _get_video_start(self, video_path, start):
        start_seek = start
        cmd = (
            ffmpeg
                .input(video_path, ss=start_seek, t=self.num_sec + 0.1)
                .filter('fps', fps=self.fps)
        )
        if self.center_crop:
            aw, ah = 0.5, 0.5
        else:
            aw, ah = random.uniform(0, 1), random.uniform(0, 1)
        if self.crop_only:
            cmd = (
                cmd.crop('(iw - {})*{}'.format(self.size, aw),
                         '(ih - {})*{}'.format(self.size, ah),
                         str(self.size), str(self.size))
            )
        else:
            cmd = (
                cmd.crop('(iw - min(iw,ih))*{}'.format(aw),
                         '(ih - min(iw,ih))*{}'.format(ah),
                         'min(iw,ih)',
                         'min(iw,ih)')
                    .filter('scale', self.size, self.size)
            )
        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, quiet=True)
        )
        video = np.frombuffer(out, np.uint8).reshape([-1, self.size, self.size, 3])
        video = th.from_numpy(np.array(video))
        video = video.permute(3, 0, 1, 2)
        if video.shape[1] < self.num_frames:
            zeros = th.zeros((3, self.num_frames - video.shape[1], self.size, self.size), dtype=th.uint8)
            video = th.cat((video, zeros), axis=1)
        return video[:, :self.num_frames]

    def _get_video_path(self, video_id):
        if os.path.isfile(os.path.join(self.videos_path, video_id + '.mp4')):
            video_path = os.path.join(self.videos_path, video_id + '.mp4')
        elif os.path.isfile(os.path.join(self.videos_path, video_id + '.mkv')):
            video_path = os.path.join(self.videos_path, video_id + '.mkv')
        elif os.path.isfile(os.path.join(self.videos_path, video_id + '.webm')):
            video_path = os.path.join(self.videos_path, video_id + '.webm')
        else:
            raise ValueError

        return video_path

    def __getitem__(self, idx):
        video_id = self.video_list['video_id'][idx]
        video_path = self._get_video_path(video_id)

        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
             video_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        video_duration = float(result.stdout)

        video = self._get_video(video_path, video_duration)

        return {'video': video, 'video_id': video_id}

