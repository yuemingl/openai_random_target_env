import numpy as np
import math
import random

class RandomTargetEnv:
    """
    A test environment. Given start point (0,0) and target point(x1, y1),
    the agent take actions 0,1,2,3 (forward, backword, turn left, turn right)
    and step length 0.1 to reach the target point
    """
    def __init__(self):
        self.num_obs = 4
        self.x_ = 0
        self.y_ = 0
        self.x1_ = random.randint(1,10)
        self.y1_ = random.randint(1,10)
        self.observation = np.zeros(self.num_obs, dtype=np.float32)

    def step(self, action):
        dx = 0
        dy = 0
        if action == 0:
            dx = 0.1
        elif action == 1:
            dx = -0.1
        elif action == 2:
            dy = 0.1
        else:
            dy = -0.1
        self.x_ += dx
        self.y_ += dy
        #print('current position:', self.x_, self.y_)
        xx = self.x1_ - self.x_
        yy = self.y1_ - self.y_
        return -math.sqrt(xx*xx + yy*yy)

    def reset(self):
        self.x_ = 0
        self.y_ = 0
        self.x1_ = random.randint(1,10)
        self.y1_ = random.randint(1,10)        
        self.observation = np.zeros(self.num_obs, dtype=np.float32)
        return self.observation

    def observe(self, ob):
        ob[0] = self.x_
        ob[1] = self.y_
        ob[2] = self.x1_
        ob[3] = self.y1_

    def isTerminalState(self):
        xx = self.x1_ - self.x_
        yy = self.y1_ - self.y_
        disToTarget = math.sqrt(xx*xx + yy*yy)
        initDis = math.sqrt(self.x1_*self.x1_ + self.y1_*self.y1_)
        if(disToTarget <= 0.01):
            terminalReward = 0.
            #print('terminate dis=',dis, 'terminalReward=', terminalReward)
            return True, terminalReward
        elif disToTarget > initDis+0.1:
            terminalReward = -initDis
            #print('terminate dis=',dis, 'terminalReward=', terminalReward)
            return True, terminalReward
        else:
            return False, 0.


