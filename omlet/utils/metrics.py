import torch


def accuracy(output, target, topk=(1,), scale_100=False):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    Accuracy is a float between 0.0 and 1.0
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        mult = 100. if scale_100 else 1.
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(float(correct_k.mul_(mult / batch_size).item()))
        return res


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name=None, formatter='.2f',
                 keep_history=False):
        """
        Args:
            keep_history: if True, keeps a list of [(value, n)]
        """
        self.name = name
        self.fmt = formatter.lstrip(':')
        self.keep_history = keep_history
        self.reset()

    def reset(self):
        self.history = []
        self.sum = 0
        self.size = 0
        self.n = 0

    def update(self, value, size=1):
        value = float(value)
        size = int(size)
        self.sum += value * size
        self.size += size
        self.n += 1
        if self.keep_history:
            self.history.append((value, size))

    @property
    def value(self):
        if self.size == 0:
            return 0.
        return self.sum / self.size

    def __float__(self):
        return float(self.value)

    def __str__(self):
        # fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        if self.name:
            fmtstr = '{name} {avg:' + self.fmt + '}'
        else:
            fmtstr = '{avg:' + self.fmt + '}'
        return fmtstr.format(name=self.name, avg=self.value)


