from funlib.geometry import Coordinate, Roi
from dacapo.experiments.architectures import CNNectomeUNetConfig
from dacapo.experiments.tasks import DistanceTaskConfig

channels = [
    "ecs", # extra cellular space
    "plasma_membrane",
    "mito",
    "mito_membrane",
    "vesicle",
    "vesicle_membrane",
    "mvb", # endosomes
    "mvb_membrane",
    "er",
    "er_membrane",
    "eres",
    "nucleus",
    "microtubules",
    "microtubules_out",
]

architecture_config = CNNectomeUNetConfig(
    name="CellMapArchitecture",
    input_shape=Coordinate(216, 216, 216), # can be changed
    eval_shape_increase=Coordinate(72, 72, 72), # can be changed
    fmaps_in=1,
    num_fmaps=12,
    fmaps_out=72,
    fmap_inc_factor=6,
    downsample_factors=[(2, 2, 2), (3, 3, 3), (3, 3, 3)],
    constant_upsample=True,
    upsample_factors=[(2, 2, 2)],
)

task_config = DistanceTaskConfig(
    name="DistancePrediction",
    # important
    channels=channels,
    scale_factor=50, # target = tanh(distance / scale)
    # training
    mask_distances=True,
    # evaluation
    clip_distance=50,
    tol_distance=10,
)