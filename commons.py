from copy import deepcopy
import torch 
from torch import nn

class CommonFunctions:
    def __init__(self):
        self.writer = None
        self.env = None
        # is share_memory the replay buffer? - yes, reimplement this
        self.shared_memory = None
        self.shared_weights = dict()
        self.chronic_priority = None
        # learning_end = 0 -> learning has not ended yet
        self.learnings_num = None
        self.memory = None
        self.nb_frame = 100000
        
        self.log_alpha = None
        self.disturbance = 0
        self.emb = nn.Sequential()
        self.Q = nn.Sequential()
        self.actor = nn.Sequential()

    # TODO!!!: save_weights() in learner class or here?
        
    def load_weights(self):
        try:
            self.emb.load_state_dict(self.shared_weights['emb'])
            self.Q.load_state_dict(self.shared_weights['Q'])
            self.actor.load_state_dict(
                self.shared_weights['actor'])
            self.log_alpha = torch.tensor(
                self.shared_weights['log_alpha'], device=self.device)
            self.disturbance = torch.tensor(
                self.shared_weights['jacob_disturbance'], device=self.device)
            return True
        except KeyError:
            return False
        
    def save_weights(self):
        self.shared_weights['emb'] = deepcopy(
            self.emb).cpu().state_dict()
        self.shared_weights['Q'] = deepcopy(
            self.Q).cpu().state_dict()
        self.shared_weights['actor'] = deepcopy(
            self.actor).cpu().state_dict()
        self.shared_weights['log_alpha'] = self.log_alpha.clone().detach().item()
        self.shared_weights['jacob_disturbance'] = self.disturbance.clone().detach().item()
        

    def __del__(self):
        self.writer.close()
        self.env.close()

    def load_memory(self):
        while not self.shared_memory.empty():
            batch = self.shared_memory.get()
            self.memory.append(batch)

    def append_shared_sample(self, s, m, a, r, s2, m2, d, order):
        if self.use_order:
            self.shared_memory.put((s, m, a, r, s2, m2, int(d), order))
        else:
            self.shared_memory.put((s, m, a, r, s2, m2, int(d)))

    def save_chronic_priority(self, arr):
        self.chronic_priority = arr
    
    def load_chronic_priority(self):
        return self.chronic_priority
    
    def incr_learnings_num(self):
        self.learnings_num[0] += 1

    def load_learning_end_flag(self):
        if self.learnings_num[0] < self.nb_frame:
            return True
        else:
            return False
