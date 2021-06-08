# Copyright (c) Open-MMLab. All rights reserved.
import os
import os.path as osp
from collections import OrderedDict

import cv2
import numpy as np
from cv2 import (CAP_PROP_FOURCC, CAP_PROP_FPS, CAP_PROP_FRAME_COUNT,
                 CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH,
                 CAP_PROP_POS_FRAMES)
from tqdm import tqdm

from mmcv.utils import (check_file_exist, mkdir_or_exist, track_progress)


class Cache:

    def __init__(self, capacity):
        self._cache = OrderedDict()
        self._capacity = int(capacity)
        if capacity <= 0:
            raise ValueError('capacity must be a positive integer')

    @property
    def capacity(self):
        return self._capacity

    @property
    def size(self):
        return len(self._cache)

    def put(self, key, val):
        if key in self._cache:
            return
        if len(self._cache) >= self.capacity:
            self._cache.popitem(last=False)
        self._cache[key] = val

    def get(self, key, default=None):
        val = self._cache[key] if key in self._cache else default
        return val


class VideoReader:
    """Video class with similar usage to a list object.

    This video warpper class provides convenient apis to access frames.
    There exists an issue of OpenCV's VideoCapture class that jumping to a
    certain frame may be inaccurate. It is fixed in this class by checking
    the position after jumping each time.
    Cache is used when decoding videos. So if the same frame is visited for
    the second time, there is no need to decode again if it is stored in the
    cache.

    :Example:

    >>> v = VideoReader('sample.mp4')
    >>> len(v)  # get the total frame number with `len()`
    120
    >>> v[5]  # get the 6th frame
    """

    def __init__(self, filename, cache_capacity=10):
        # Check whether the video path is a url
        if not filename.startswith(('https://', 'http://')):
            check_file_exist(filename, 'Video file not found: ' + filename)
        self._vcap = cv2.VideoCapture(filename)
        assert cache_capacity > 0
        self._cache = Cache(cache_capacity)
        self._position = 0
        # get basic info
        self._width = int(self._vcap.get(CAP_PROP_FRAME_WIDTH))
        self._height = int(self._vcap.get(CAP_PROP_FRAME_HEIGHT))
        self._fps = self._vcap.get(CAP_PROP_FPS)
        self._frame_cnt = int(self._vcap.get(CAP_PROP_FRAME_COUNT))
        self._fourcc = self._vcap.get(CAP_PROP_FOURCC)

    @property
    def vcap(self):
        """:obj:`cv2.VideoCapture`: The raw VideoCapture object."""
        return self._vcap

    @property
    def opened(self):
        """bool: Indicate whether the video is opened."""
        return self._vcap.isOpened()

    @property
    def width(self):
        """int: Width of video frames."""
        return self._width

    @property
    def height(self):
        """int: Height of video frames."""
        return self._height

    @property
    def resolution(self):
        """tuple: Video resolution (width, height)."""
        return (self._width, self._height)

    @property
    def fps(self):
        """float: FPS of the video."""
        return self._fps

    @property
    def frame_cnt(self):
        """int: Total frames of the video."""
        return self._frame_cnt

    @property
    def fourcc(self):
        """str: "Four character code" of the video."""
        return self._fourcc

    @property
    def position(self):
        """int: Current cursor position, indicating frame decoded."""
        return self._position

    def _get_real_position(self):
        return int(round(self._vcap.get(CAP_PROP_POS_FRAMES)))

    def _set_real_position(self, frame_id):
        self._vcap.set(CAP_PROP_POS_FRAMES, frame_id)
        pos = self._get_real_position()
        for _ in range(frame_id - pos):
            self._vcap.read()
        self._position = frame_id

    def read(self):
        """Read the next frame.

        If the next frame have been decoded before and in the cache, then
        return it directly, otherwise decode, cache and return it.

        Returns:
            ndarray or None: Return the frame if successful, otherwise None.
        """
        # pos = self._position
        if self._cache:
            img = self._cache.get(self._position)
            if img is not None:
                ret = True
            else:
                if self._position != self._get_real_position():
                    self._set_real_position(self._position)
                ret, img = self._vcap.read()
                if ret:
                    self._cache.put(self._position, img)
        else:
            ret, img = self._vcap.read()
        if ret:
            self._position += 1
        return img

    def get_frame(self, frame_id):
        """Get frame by index.

        Args:
            frame_id (int): Index of the expected frame, 0-based.

        Returns:
            ndarray or None: Return the frame if successful, otherwise None.
        """
        if frame_id < 0 or frame_id >= self._frame_cnt:
            raise IndexError(
                f'"frame_id" must be between 0 and {self._frame_cnt - 1}')
        if frame_id == self._position:
            return self.read()
        if self._cache:
            img = self._cache.get(frame_id)
            if img is not None:
                self._position = frame_id + 1
                return img
        self._set_real_position(frame_id)
        ret, img = self._vcap.read()
        if ret:
            if self._cache:
                self._cache.put(self._position, img)
            self._position += 1
        return img

    def current_frame(self):
        """Get the current frame (frame that is just visited).

        Returns:
            ndarray or None: If the video is fresh, return None, otherwise
                return the frame.
        """
        if self._position == 0:
            return None
        return self._cache.get(self._position - 1)

    def cvt2frames(self,
                   frame_dir,
                   file_start=0,
                   filename_tmpl='{:06d}.jpg',
                   start=0,
                   max_num=0,
                   show_progress=True):
        """Convert a video to frame images.

        Args:
            frame_dir (str): Output directory to store all the frame images.
            file_start (int): Filenames will start from the specified number.
            filename_tmpl (str): Filename template with the index as the
                placeholder.
            start (int): The starting frame index.
            max_num (int): Maximum number of frames to be written.
            show_progress (bool): Whether to show a progress bar.
        """
        mkdir_or_exist(frame_dir)
        if max_num == 0:
            task_num = self.frame_cnt - start
        else:
            task_num = min(self.frame_cnt - start, max_num)
        if task_num <= 0:
            raise ValueError('start must be less than total frame number')
        if start > 0:
            self._set_real_position(start)

        def write_frame(file_idx):
            img = self.read()
            filename = osp.join(frame_dir, filename_tmpl.format(file_idx))
            cv2.imwrite(filename, img)

        if show_progress:
            track_progress(write_frame, range(file_start,
                                              file_start + task_num))
        else:
            for i in range(task_num):
                img = self.read()
                if img is None:
                    break
                filename = osp.join(frame_dir,
                                    filename_tmpl.format(i + file_start))
                cv2.imwrite(filename, img)

    def __len__(self):
        return self.frame_cnt

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [
                self.get_frame(i)
                for i in range(*index.indices(self.frame_cnt))
            ]
        # support negative indexing
        if index < 0:
            index += self.frame_cnt
            if index < 0:
                raise IndexError('index out of range')
        return self.get_frame(index)

    def __iter__(self):
        self._set_real_position(0)
        return self

    def __next__(self):
        img = self.read()
        if img is not None:
            return img
        else:
            raise StopIteration

    next = __next__

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._vcap.release()


camvid_path = "./data/camvid"   # put the downloaded four videos on /data/camvid/raw .
video_names = ['0005VD.MXF', '0006R0.MXF', '0016E5.MXF', '01TP_extract.avi']
filename_tmpl = ['Seq05VD_f{:05d}.png', '0006R0_f{:05d}.png', '0016E5_{:05d}.png', '0001TP_{:06d}.png']
file_start = [-1, -1, -1, 6660]
start = [0, 0, 0, 0]  # read frames: 6600, 7170, 11382, 3751; soft frames:6599, 7169, 11381, 3747
for i, video in enumerate(video_names):
    # if video == '01TP_extract.avi':
    video_path = osp.join(camvid_path, 'raw', video)
    video_reader = VideoReader(video_path)
    print('video name:', video)
    print('video:info')
    print('opened:', video_reader.opened, 'resolution', video_reader.resolution, 'fps:', video_reader.fps,
          'total frames:', video_reader.frame_cnt)
    print('\n')
    save_path = osp.join(camvid_path, 'image_sequence')
    if osp.exists(save_path):
        shutil.rmtree(save_path)

    video_reader.cvt2frames(frame_dir=save_path, file_start=file_start[i], filename_tmpl=filename_tmpl[i],
                            start=start[i])     # extract frames from the given video