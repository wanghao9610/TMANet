_base_ = [
    '../../_base_/models/tmanet_r50-d8.py', '../../_base_/datasets/cityscapes_video_769x769.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_80k.py'
]
model = dict(
    decode_head=dict(type='TMAHead', align_corners=True, sequence_num=2, key_channels=64, value_channels=256),
    auxiliary_head=dict(align_corners=True))
data = dict(train=dict(sequence_range=2,
                       sequence_num=2),
            val=dict(sequence_range=2,
                     sequence_num=2),
            test=dict(sequence_range=2,
                      sequence_num=2))
test_cfg = dict(mode='slide', crop_size=(769, 769), stride=(513, 513))
