import math
from multiprocessing import Pool
from typing import List, Callable, Any
from tqdm import tqdm


def build_check(pid:int, datas:List, check_unit:Callable[[Any], bool]):
    def __pbar(pid, total, desc='POOL', colour='cyan'):
        return tqdm(total=total, desc=desc, colour=colour, disable=pid!=0)
    res = []
    n = len(datas)
    pbar = __pbar(pid, n)
    for data in datas:
        if check_unit(data):
            res.append(data)
        pbar.update()

    return res


class PoolData:
    def __init__(self, datas:List, nproc):
        if nproc == -1:
            self.nproc = len(datas)
            self.data = datas
        else:
            self.nproc = nproc
            self.data = []
            ndata = len(datas)
            step = math.ceil(ndata / nproc)
            b = 0
            for i in range(nproc):
                e = min((i+1)*step, ndata) 
                self.data.append(datas[b:e])
                b = e
        self.run_end = False
    
    def run(self, check_unit:Callable[[Any], bool]):
        rets = []
        pool = Pool(self.nproc)
        for i in range(self.nproc):
            res = pool.apply_async(func=build_check, args=(i, self.data[i], check_unit))
            rets.append(res)

        self.datas = []
        for r in rets:
            r.wait()
            self.datas.extend(r.get())

        pool.close()
        pool.join()

        self.run_end = True
        return self.datas
    
    def __getitem__(self, index):
        if self.run_end:
            return self.datas[index]
        else:
            return None
