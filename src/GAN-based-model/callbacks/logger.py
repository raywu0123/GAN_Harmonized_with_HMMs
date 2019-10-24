from tensorboardX import SummaryWriter


class Logger:

    def __init__(self, print_step=1, decay_rate=0.9, tensorboard=True, logdir=None):
        self.print_step = print_step
        self.decay_rate = decay_rate
        self._step = 1
        self.values = {}
        if tensorboard:
            self.writer = SummaryWriter(logdir=logdir)
        else:
            self.writer = None

    def update(self, items: dict, ema=True):
        for key, val in items.items():
            if key not in self.values or ema:
                self.values[key] = val
            else:
                try:
                    self.values[key] = self.values[key] * self.decay_rate + val * (1 - self.decay_rate)
                except TypeError:
                    self.values[key] = val

    def step(self):
        if self._step % self.print_step == 0:
            self.print_log()
        if self.writer is not None:
            self.log_tensorboard()
        self._step += 1

    def print_log(self):
        print(f'Step: {self._step:5d} ')
        for key, val in self.values.items():
            if isinstance(val, int):
                print(f'  {key}: {val:5d}')
            elif isinstance(val, float):
                print(f'  {key}: {val:.4f}')
            else:
                print(f'  {key}: {val}')

    def log_tensorboard(self):
        for key, val in self.values.items():
            if isinstance(val, (int, float)):
                self.writer.add_scalar(key, val, global_step=self._step)
