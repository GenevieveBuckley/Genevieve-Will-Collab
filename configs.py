from funlib.geometry import Coordinate, Roi
from dacapo.experiments.architectures import CNNectomeUNetConfig
from dacapo.experiments.tasks import DistanceTaskConfig

from skimage.util import img_as_float
import tifffile
import torch

input_voxel_size = (8, 8, 8)
output_voxel_size = (4, 4, 4)

channels = [
    "ecs",  # extra cellular space
    "plasma_membrane",
    "mito",
    "mito_membrane",
    "vesicle",
    "vesicle_membrane",
    "mvb",  # endosomes
    "mvb_membrane",
    "er",
    "er_membrane",
    "eres",
    "nucleus",
    "microtubules",
    "microtubules_out",
]

device = 'cpu'

architecture_config = CNNectomeUNetConfig(
    name="CellMapArchitecture",
    input_shape=Coordinate(216, 216, 216),  # can be changed (output size approx 96x96x96)
    eval_shape_increase=Coordinate(72, 72, 72),  # can be changed (output size approx 168x168x168)
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
    scale_factor=50,  # target = tanh(distance / scale)
    # training
    mask_distances=True,
    # evaluation
    clip_distance=50,
    tol_distance=10,
)

# create backbone from config
architecture = architecture_config.architecture_type(architecture_config)

# initialize task from config
task = task_config.task_type(task_config)

# adding final layers/activations to create the model
model = task.create_model(architecture)

# # load weights from file
weights_file = "CellmapModelWeights/modelA"
weights = torch.load(weights_file, map_location=device)  # map_locations options are "cpu" or "cuda" (this is becuase when pytorch saves weights, it also remembers what device it was on)

# # initialize model weights
model.load_state_dict(weights['model'])

# load data
filename = "/Volumes/GENEVIEVE/data/janelia/jrc_hela-2_tinycrop.tif"
image = img_as_float(tifffile.imread(filename))
print(image.shape)

# select a small region of raw that can be passed through the model:
image = torch.from_numpy(image).to(device).float()
image = image.unsqueeze(0).unsqueeze(0)

model.to(device)

prediction = model(image)
