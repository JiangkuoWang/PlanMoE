import os
import sys


def main():
    host_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    host_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    local_size = int(os.environ.get('LOCAL_SIZE', 1))

    master_addr = os.environ['MASTER_ADDR'] if host_size > 1 else '127.0.0.1'
    master_port = int(os.environ.get('MASTER_PORT', 23232))
    if 'OMP_NUM_THREADS' not in os.environ:
        os.environ['OMP_NUM_THREADS'] = str(local_size)
    cmd_args = [sys.executable, '-m'] + ['torch.distributed.launch',] + [
        '--nproc_per_node=%d' % local_size,
        '--nnodes=%d' % host_size,
        '--node_rank=%d' % host_rank,
        '--master_addr=%s' % master_addr,
        '--master_port=%s' % master_port] + sys.argv[1:]

    os.execl(cmd_args[0], *cmd_args)
    

if __name__ == "__main__":
    main()
