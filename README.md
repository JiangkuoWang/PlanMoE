![image](https://github.com/user-attachments/assets/fb0425a4-db58-4b98-998c-f2f8df8366f5)

# PlanMoE
Large language models (LLMs) have achieved significant breakthroughs in many neural language processing (NLP) and computer vision tasks with increasing model sizes.  However, scaling LLMs requires a linear increase of compute with respect to the model size.  Recently, the sparsely activated mixture-of-experts (MOE) technology, which was first proposed in 1990s, has been integrated into LLMs to scale the model size to trillions of parameters with requiring only a sub-linear increase of computations.    

However, when training MoE models on a large-scale GPU/TPU cluster, it would introduce critical performance issues that make the distributed training system scale badly.  Specifically, in training MoE models, the input data (e.g., tokens) of MoE layers should be dynamically (every mini-batch) routed to different experts for computation, but the experts may be located on different workers when one worker (e.g., GPU) cannot store all experts.    

# Our Contributions
In this work, we propose PlanMoE, an extensible and efficient MoE training system, which is equipped with several features:

PlanMoE provides a generic scheduling framework that allows the communication and computation tasks in training MoE models to be scheduled in an optimal way.    
PlanMoE integrates our proposed novel all-to-all collective which better utilizes intra- and inter-connect bandwidths.    
PlanMoE supports easy extensions of customized all-to-all collectives and data compression approaches while enjoying our scheduling algorithm.    
Code Design
The PlanMoE system is designed to be extensible and efficient. To this end, we have made the following design decisions:

We modularize the time-consuming operations including data compression (a computing task), collective communication (a communication task), and expert computation (a computing task) so that these operations are easily customized with newly designed implementations.    
Based on the modularized operations, we propose an adaptive optimal scheduling algorithm to pipeline the communication and computing tasks to improve the training efficiency.    
We design a novel all-to-all algorithm, Pipe-A2A, that pipelines the intra-node communications and inter-node communications such that the intra-node bandwidth and inter-node bandwidth can be simultaneously utilized to improve communication efficiency.    




The development of this code refers to [tutel](https://github.com/microsoft/tutel).

## Prerequisite

torch>=1.9.1

## How to install

```Shell
# Install zfp
cd zfp
mkdir build
cd build
cmake ..
cmake --build . --config Release
ctest
cd ../..


cd PlanMoE
# May change include_dirs and library_dirs in setup.py
pip install -e .
```

## How to Use

```python3
# Single Machine:
 python3 -m torch.distributed.run --nproc_per_node=4 -m planmoe.examples.pre_test --batch_size=16
# Distribute:
# pls refers to planmoe/examples/run_mpi.sh
```

## How to Add a New Compressor

1. Navigate to the planmoe/custom/compressor/ directory.

2. Create a new compressor class that inherits from the AbstractCompressor class.

3. Implement the virtual functions defined in abstract.h within your new compressor class.

## How to Add a New AllToAll Communication Algorithm

1. Navigate to the planmoe/custom/comm/ directory.

2. Create a new comm class that inherits from the AbstractComm class.

3. Implement the virtual functions defined in abstract.h within your new comm class.

## Test Environment

- g++==7.5.0
- cuda==10.2
- gpu==2080Ti


