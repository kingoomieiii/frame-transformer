from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class PolynomialDecayScheduler(_LRScheduler):
    def __init__(self, optimizer, target=1e-8, power=1.0, num_decay_steps=120000, start_step=16000, current_step=0, verbose_skip_steps=1000):
        self.start_step = start_step
        self.current_step_init = current_step
        self.current_step = current_step
        self.verbose_skip_steps = verbose_skip_steps
        self.num_decay_steps = num_decay_steps
        self.target = target
        self.power = power
        self.last_step = 0

        self.target_lr = target
        self.current_lr = optimizer.param_groups
        
        super().__init__(optimizer)    
        
    # def step(self, step=None):
    #     if step is None:
    #         step = self.last_step + 1
    #     self.last_step = step if step != 0 else 1
    #     for i, param_group in enumerate(self.optimizer.param_groups):
    #         if self.last_step <= self.num_decay_steps:
    #             decay_lrs = [(base_lr - self.target) * 
    #                         ((1 - self.last_step / self.num_decay_steps) ** (self.power)) + 
    #                         self.target for base_lr in self.base_lrs]
    #             for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
    #                 param_group['lr'] = lr
    #                 print(lr)

    def step(self):
        if self.current_step > self.start_step and self.current_step < (self.start_step + self.num_decay_steps + 1):
            if self.current_step == self.start_step + 1:
                self.current_lr = self.optimizer.param_groups            
            
            for i, param_group in enumerate(self.optimizer.param_groups):
                self.current_lr[i]['lr'] =  (self.base_lrs[i] - self.target) * ((1 - (self.current_step - self.start_step) / (self.num_decay_steps - self.start_step)) ** self.power) + self.target
                
                param_group['lr'] = self.current_lr[i]['lr']
                if self.current_step % self.verbose_skip_steps == 0:
                    print(' Step {:5d} of {:5d}: decreased learning rate'
                            ' of group {} to {:.4e}.'.format(self.current_step, self.start_step + self.num_decay_steps, i, self.current_lr[i]['lr']))

        if self.current_step < self.start_step + self.num_decay_steps + 1:
            self.current_step = self.current_step + 1

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)