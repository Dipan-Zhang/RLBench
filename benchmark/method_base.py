"""
a abstract of different methods =>
- input: a dict of data 
- output: trajectory
- method: RAM, RobABC VRB etc
"""

from abc import ABC, abstractmethod


class AffordanceMethodBase(ABC):
    @abstractmethod
    def __init__(self, data):
        pass

    @abstractmethod
    def get_trajectory(self):
        pass
    
