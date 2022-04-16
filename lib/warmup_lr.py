from torch.optim import Optimizer

class WarmupLR(object):
    def __init__(self, optimizer, target_lr=1e-3, num_steps=16000, current_step=0, verbose=False, verbose_skip_steps=1000):
        self.target_lr = target_lr
        self.num_steps = num_steps
        self.current_step = current_step
        self.verbose = verbose
        self.verbose_skip_steps = verbose_skip_steps

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))

        self.optimizer = optimizer
        self._reset()

    def _reset(self):
        self.current_step = 1
        self.current_lr = (self.current_step) * (self.target_lr / self.num_steps)
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.current_lr
            if self.verbose and self.current_step % self.verbose_skip_steps == 0:
                print(' Step {:5d}: increased learning rate'
                        ' of group {} to {:.4e}.'.format(self.current_step, i, self.current_lr))

    def step(self):
        if self.current_step < self.num_steps:
            self.current_step = self.current_step + 1
            self.current_lr = (self.current_step) * (self.target_lr / self.num_steps)

            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.current_lr
                if self.verbose and self.current_step % self.verbose_skip_steps == 0:
                    print(' Step {:5d}: increased learning rate'
                            ' of group {} to {:.4e}.'.format(self.current_step, i, self.current_lr))

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)