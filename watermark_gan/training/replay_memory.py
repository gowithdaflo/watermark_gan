import torch
import random
        
class ReplayMemory:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.cur_length = 0
        self.data = None
        self.indices = set()
        
    def __len__(self):
        return self.cur_length
        
    def add(self, batch: dict) -> None:
        if self.data is None:
            # first batch
            self.data = {k: torch.zeros(self.capacity,*v.shape[1:]) for k,v in batch.items()}
        if self.cur_length>=self.capacity:
            return

        # copy batch to memory
        batch_size = batch[list(batch.keys())[0]].shape[0]
        s, e = self.cur_length, min(self.cur_length+batch_size, self.capacity)
        for k,v in batch.items():
            self.data[k][s:e] = v.cpu() # map to cpu to reduce gpu footprint
            
        self.cur_length = e
        
    def get(self, indices):
        batch = {}
        for k, v in self.data.items():
            batch[k] = v[indices]
        return batch
    
    def sample(self, nSamples, replace=False):
        if replace:
            idx = random.sample(range(self.cur_length), k=nSamples)
        else:
            if len(self.indices) < nSamples:
                self.indices = set(range(self.cur_length))
            
            idx = random.sample(self.indices, k=nSamples)
            for i in idx:
                self.indices.remove(i)
        
        return self.get(idx)
    
    
# memory = ReplayMemory()

# batch = {"a": torch.ones(10,5,5,5),
#         "b": torch.ones(10,7,3,3),
#         "c": torch.ones(10,6,4,4),}
        
        
# memory.add(batch)
# memory.add(batch)

# out = memory.sample(15)
# out = memory.sample(15)


# for k,v in out.items():
#     print(k, v.shape)