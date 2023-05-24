import numpy as np
import torch

def minibatch_generator(batch_size, task=None, num_rows=None):
    if isinstance(task, torch.Tensor):
        def fun():
            task = task.numpy()
            task_ids = np.unique(task)
            mbs = np.random.choice(np.arange(len(task_ids)), size=len(task_ids), replace=False)
            mbs = [task_ids[mbs[i:i + batch_size]] for i in range(0, len(mbs), batch_size)]
            mbs = [np.where(np.isin(task, mb))[0] for mb in mbs]
            return mbs
    elif isinstance(num_rows, int):
        def fun():
            temp = np.random.choice(np.arange(num_rows), size=num_rows, replace=False)
            mbs = [temp[i:i + batch_size] for i in range(0, len(temp), batch_size)]
            return mbs
    else:
        raise ValueError("Either task or num_rows argument must be provided")

    return fun