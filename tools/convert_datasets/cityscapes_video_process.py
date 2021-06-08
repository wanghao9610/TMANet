import os
import os.path as osp
splits = ['train', 'val', 'test']

# count and check the frames of cityscapes.
citys_path = '/data/datasets/video_ss/cityscapes/leftImg8bit_sequence'
for split in splits:
    sequence_path = osp.join(citys_path, split)
    citys_frames_list = {}
    for looproot, _, filenames in os.walk(sequence_path):
        for filename in filenames:
            if filename.endswith('.png'):
                dirpath = looproot.split('/')[-1]
                if dirpath not in citys_frames_list.keys():
                    citys_frames_list[dirpath] = []
                citys_frames_list[dirpath].append(filename)
    print(split)
    total_frames = 0
    for citys, frames_list in citys_frames_list.items():
        print('{:<20s}: {:<10d}'.format(citys, len(frames_list)))
        total_frames += len(frames_list)
    print('{:<20s}: {:<10d}'.format(f'{split}_total\n', total_frames))