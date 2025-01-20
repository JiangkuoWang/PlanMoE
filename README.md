# ScheMoE

ScheMoE

The development of this code refers to [tutel](https://github.com/microsoft/tutel).

## Prerequisite

torch>=1.9.1

## How to install

```Shell
# Install zfp
git clone https://github.com/Fragile-azalea/zfp.git
cd zfp
mkdir build
cd build
cmake ..
cmake --build . --config Release
ctest
cd ../..

git clone https://github.com/Fragile-azalea/ScheMoE.git
cd ScheMoE
# May change include_dirs and library_dirs in setup.py
pip install -e .
```

## How to Use

```python3
# Single Machine:
 python3 -m torch.distributed.run --nproc_per_node=4 -m schemoe.examples.pre_test --batch_size=16
# Distribute:
# pls refers to schemoe/examples/run_mpi.sh
```

## How to Add a New Compressor

1. Navigate to the schemoe/custom/compressor/ directory.

2. Create a new compressor class that inherits from the AbstractCompressor class.

3. Implement the virtual functions defined in abstract.h within your new compressor class.

## How to Add a New AllToAll Communication Algorithm

1. Navigate to the schemoe/custom/comm/ directory.

2. Create a new comm class that inherits from the AbstractComm class.

3. Implement the virtual functions defined in abstract.h within your new comm class.

## Test Environment

- g++==7.5.0
- cuda==10.2
- gpu==2080Ti

## Paper Link

https://dl.acm.org/doi/10.1145/3627703.3650083
