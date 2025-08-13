
import time

class MCTSMemo:
    def __init__(self):
        self.search_start = 0
        self.search_sum = 0
        
        self.select_start = 0
        self.select_sum = 0
        
        self.expand_start = 0
        self.expand_sum = 0
        
        self.depth_max = 0
        
    def reset(self):
        self.search_start = 0
        self.search_sum = 0
        
        self.select_start = 0
        self.select_sum = 0
        
        self.expand_start = 0
        self.expand_sum = 0
        
        self.depth_max = 0
        
    def __TimerStart(self, start) -> None:
        start = time.time()
        
    def __TimerEnd(self, start, sum) -> None:
        t = time.time() - start
        sum += t
        
    def SearchStart(self):
        self.search_start = time.time()
        
    def SearchEnd(self) -> float:
        t = time.time() - self.search_start
        self.search_sum += t
        return t
    
    def SelectStart(self):
        self.select_start = time.time()
        
    def SelectEnd(self) -> float:
        t = time.time() - self.select_start
        self.select_sum += t
        return t
    
    def ExpandStart(self):
        self.expand_start = time.time()
        
    def ExpandEnd(self) -> float:
        t = time.time() - self.expand_start
        self.expand_sum += t
        return t
    
    def depth_max_set(self, depth: int) -> None:
        if(self.depth_max < depth):
            self.depth_max = depth
        
    def PrintMemo(self) -> None:
        print(f"search_time: {self.search_sum:.4f}s")
        print(f"expand_time: {self.expand_sum:.4f}s, select_time: {self.select_sum:.4f}s")
        print(f"max_depth: {self.depth_max}")
    
    