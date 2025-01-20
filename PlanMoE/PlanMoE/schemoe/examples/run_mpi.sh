PYTHON=/home/xinglinpan/miniconda3/envs/eurosys2024fall/bin/python
LD_LIBRARY_PATH="/home/xinglinpan/nccl_2.12.12-1+cuda10.2_x86_64/lib:/home/xinglinpan/zfp/build/lib:/usr/local/cuda-10.2/lib64/"

NNODES=${#ADDR_LIST[@]}
MASTER_ADDR=${ADDR_LIST[0]}

for a2a_ffn_overlap_degree in 2; do
        mpiexec -x PATH=$PATH -x CUDA_HOME=/usr/local/cuda-10.2/ -x NCCL_DEBUG=WARN -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH -x MASTER_ADDR=ethgpu9 -x LOCAL_SIZE=4 --prefix /home/xinglinpan/mpi/openmpi-4.1.4/ --host ethgpu9,ethgpu10 -bind-to none $PYTHON launch.py pre_test.py --a2a_ffn_overlap_degree=$a2a_ffn_overlap_degree --log='test.log' --encode='no'
        sleep 5s
done
