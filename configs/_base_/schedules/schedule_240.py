# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=True)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=240)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=20)
evaluation = dict(by_epoch=False, interval=10000, metric='mIoU')
