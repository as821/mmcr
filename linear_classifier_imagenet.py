import sys

sys.path.append("..")

import submitit
from mmcr.imagenet.train_linear_classifier import train_classifier
from torch import nn

# submitit stuff
slurm_folder = "./slurm/classifier/tmp/%j"


executor = submitit.AutoExecutor(folder=slurm_folder)
executor.update_parameters(mem_gb=128, timeout_min=10000)
executor.update_parameters(slurm_array_parallelism=1024)
executor.update_parameters(gpus_per_node=1)
executor.update_parameters(cpus_per_task=13)
executor.update_parameters(slurm_partition="gpu")
executor.update_parameters(constraint="a100-80gb")
executor.update_parameters(name="classifier_train")

job = executor.submit(
    train_classifier,
    model_path="./training_checkpoints/imagenet/eight_view_two/latest-rank0",
    batch_size=2048,
    lr=0.3,
    epochs=100,
    #save_path='./training_checkpoints/imagenet/2_900/',
    #save_name='classifier'
)