import sys
import threading
import torch
import multiprocessing as mp
import my_project.utils.logger_cfg as logger_cfg
logger = logger_cfg.get_logger()

def start_processes(start_funcs=[], start_args=[]):
    assert len(start_funcs) == len(start_args)
    pipe_readers = []
    processes = []
    for start_func, start_arg in zip(start_funcs, start_args):
        pipe_reader, pipe_writer = mp.Pipe(duplex=False)
        process = mp.Process(
            target=start_func,
            args=start_arg + (pipe_writer,),
        )
        process.start()
        pipe_readers.append(pipe_reader)
        processes.append(process)
    
    # wait to ready
    for index, pipe_reader in enumerate(pipe_readers):
        init_state = pipe_reader.recv()
        if init_state != 'init ok':
            logger.error(f"init func {start_funcs[index].__name__} : {str(init_state)}")
            for proc in processes:
                proc.kill()
            sys.exit(1)
        else:
            logger.info(f"init func {start_funcs[index].__name__} : {str(init_state)}")
    
    assert all([proc.is_alive() for proc in processes]), f"{proc} process is not alive"
    return

class ReqIdGenerator:
    def __init__(self, start=0):
        self.current_id = start
        self.lock = threading.Lock()

    def generate_ids(self, batch_size):
        with self.lock:
            id = self.current_id
            self.current_id += 1

        ids = torch.IntTensor([(id, ind) for ind in range(batch_size)])
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device="cpu")
        return ids, unfinished_sequences