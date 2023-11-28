import torch
from triton_nautilus.export import main as export 
import subprocess
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import s3fs

if TYPE_CHECKING:
    import hermes.quiver as qv

def main(
    weights: str = "s3://triton-nautilus/model.pt",
    repository_directory: str = "/tmp/model_repo",
    logdir: str = "/tmp/model_repo/logs",
    num_ifos: int = 2,
    kernel_length: float = 16,
    inference_sampling_rate: float = 4,
    sample_rate: float = 2048,
    batch_size: int = 512,
    fduration: float = 2,
    psd_length: float = 10,
    fftlength: Optional[float] = None,
    highpass: Optional[float] = None,
    streams_per_gpu: int = 1,
    aframe_instances: Optional[int] = None,
    platform: "qv.Platform" = "qv.Platform.ONNX",
    clean: bool = False,
    verbose: bool = False,
):

    # pull weights via s3
    s3 = s3fs.S3FileSystem(endpoint_url="http://rook-ceph-rgw-nautiluss3.rook")
    model_dir = Path("/tmp/model_repo")
    model_dir.mkdir(parents=True, exist_ok=True)
    local_weights = model_dir / "weights.pt"
    s3.get(weights, local_weights)


    # first export model to tensorrt, along with 
    # constructing snapshotter and whitener
    export(
        local_weights,
        repository_directory,
        logdir,
        num_ifos,
        kernel_length,
        inference_sampling_rate,
        sample_rate,
        batch_size,
        fduration,
        psd_length,
        fftlength,
        highpass,
        streams_per_gpu,
        aframe_instances,
        platform,
        clean,
        verbose,
    )

    # then run triton server
    command = (
        '/bin/bash -c '
        '"LD_LIBRARY_PATH=\\"$LD_LIBRARY_PATH:$EXTRA_NV_PATHS\\" '
        'bin/tritonserver --model-repository /home/ethanmarx/model_repo '
        '--repository-poll-secs 30 --model-control-mode explicit"'
    )
    output = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

