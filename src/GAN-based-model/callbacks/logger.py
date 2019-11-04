from tensorboardX import SummaryWriter


class Logger:

    def __init__(self, print_step=1, decay_rate=0.9, tensorboard=True, logdir=None):
        self.print_step = print_step
        self.decay_rate = decay_rate
        self._step = 1
        self.values = {}
        self.group_name = {}
        if tensorboard:
            self.writer = SummaryWriter(logdir=logdir)
        else:
            self.writer = None

    def update(self, items: dict, ema=True, group_name=None):
        for key, val in items.items():
            if key not in self.values or not ema:
                self.values[key] = val
            else:
                try:
                    self.values[key] = self.values[key] * self.decay_rate + val * (1 - self.decay_rate)
                except TypeError:
                    self.values[key] = val

            if group_name is not None:
                self.group_name[key] = group_name

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
            add_fn = None
            if isinstance(val, (int, float)):
                add_fn = self.writer.add_scalar
            elif isinstance(val, str):
                add_fn = self.writer.add_text
            else:
                continue

            if key in self.group_name:
                add_fn(f'{self.group_name[key]}/{key}', val, global_step=self._step)
            else:
                add_fn(key, val, global_step=self._step)
