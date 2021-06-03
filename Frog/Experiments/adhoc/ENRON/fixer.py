from abc import abstractmethod
from typing import List

import numpy as np


class Fixer:

    
    def __init__(self, truth: np.ndarray):
        assert len(truth.shape) == 1
        self.truth = truth
        self.deletions = []

    @abstractmethod
    def fix(self, trainer, delta: np.ndarray, rank: np.ndarray):
        pass

    def get_candidate_removal(self, rank: np.ndarray, trainer) -> int:
        # This is the ranking over the remaining indexes
        removes = np.argsort(rank)
        remaining = trainer.deltas.nonzero()[0]
        removes   = remaining[removes]
        
        # assume_unique is required otherwise setdiff1d will sort the array
        removes = np.setdiff1d(removes, self.deletions, assume_unique=True)  
        to_remove = removes[0]
        self.deletions.append(to_remove)
        return to_remove

    def get_deletions(self) -> np.ndarray:
        return np.asarray(self.deletions)
    
    @abstractmethod
    def name(self):
        pass


class AutoFixer(Fixer):
    
    def fix(self, trainer , rank: np.ndarray):
        to_remove = self.get_candidate_removal(rank, trainer)
        trainer.remove(to_remove)
        return self.truth[to_remove]
    
    def name(self):
        return "AutoFixer"
        
class HumanFixer(Fixer):
    
    def fix(self, trainer , rank: np.ndarray):
        to_remove = self.get_candidate_removal(rank, trainer)

        if self.truth[to_remove]:
            trainer.remove(to_remove)
        
        return self.truth[to_remove]
    
    def name(self):
        return "HumanFixer"

class OracleFixer(Fixer):
    
    def fix(self, trainer , rank: np.ndarray):
        to_remove = self.get_candidate_removal(rank, trainer)

        if self.truth[to_remove]:
            trainer.correct_label(to_remove)
        
        return self.truth[to_remove]
    
    def name(self):
        return "OracleFixer"
