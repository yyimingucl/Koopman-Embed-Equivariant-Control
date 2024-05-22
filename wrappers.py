"""
Filename: wrappers.py
Created: 24/09/2023
Description:
    This file contains environment wrapper for pendulum environments.
"""

from gymnasium import ObservationWrapper
from typing import Any, TypeVar
import numpy as np

WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")

class modified_pendulum(ObservationWrapper):
    def __init__(self, env):
        
        super(modified_pendulum, self).__init__(env)
    def reset(
        self, *, init_state=None, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[WrapperObsType, dict[str, Any]]:
        """Modifies the :attr:`env` after calling :meth:`reset`, returning a modified observation using :meth:`self.observation`."""
        if init_state is None:
            obs, info = self.env.reset(seed=seed, options=options)
        else:
            obs, info = self.env.reset(init_state=init_state, seed=seed, options=options)
        return self.observation(obs), info
    
    def observation(self, observation: WrapperObsType) -> WrapperObsType:   
        return np.array([np.arctan2(observation[1],observation[0]), observation[2]])


