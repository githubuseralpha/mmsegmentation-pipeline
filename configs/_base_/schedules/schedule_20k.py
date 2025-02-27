# optimizer
optimizer = dict(type='AdamW', lr=0.00002, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
# param_scheduler = [
#     dict(
#         type='PolyLR',
#         eta_min=1e-4,
#         power=0.9,
#         begin=0,
#         end=50000,
#         by_epoch=False)
# ]
# training schedule for 20k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=30000, val_interval=750)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=750),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, show=False, interval=30),
    # save_prediction=dict(type='SavePredictionHook', output_dir='val_predictions_4')
    )
