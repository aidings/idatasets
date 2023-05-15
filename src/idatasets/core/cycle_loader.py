import math

class CycleLoader:
    def __init__(self, dataloader, max_iter, save_freq=None):
        dloader = self.__cycle(dataloader)
        self.diter = iter(dloader)

        if max_iter:
            save_freq = save_freq or max_iter
            n = max_iter // save_freq
            m = max_iter % save_freq
            seg_num = [save_freq] * n
            if m != 0:
                seg_num += [m]
            self.seg_num = seg_num 

    def __len__(self):
        try:
            return len(self.seg_num)
        except:
            return math.inf
    
    def size(self, idx):
        try:
            return self.seg_num[idx]
        except:
            return math.inf
    
    def step(self, idx):
        try:
            return sum(self.seg_num[:idx])
        except:
            return math.inf

    def __cycle(self, dataloader):
        while True:
            for data in dataloader:
                yield data
    
    def next(self):
        return next(self.diter) 

    def __call__(self, idx):
        for _ in range(self.seg_num[idx]):
            yield next(self.diter)
