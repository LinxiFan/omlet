import time


class Timer:
    def __init__(self):
        self._start = None
        self._end = None

    def start(self):
        self._start = time.time()

    def elapsed(self):
        self._end = time.time()
        self._interval = self._end - self._start
        return self._interval

    @staticmethod
    def pformat(interval, unit='auto', show_hms=True, float_fmt='.2f'):
        if not float_fmt:
            float_fmt = '.2f'
        if not float_fmt.startswith('{'):
            float_fmt = '{:' + float_fmt + '}'
        if unit == 'auto':
            kwargs = dict(show_hms=show_hms, float_fmt=float_fmt)
            if interval >= 0.1:
                return Timer.pformat(interval, unit='s', **kwargs)
            elif interval >= 1e-4:
                return Timer.pformat(interval, unit='ms', **kwargs)
            elif interval >= 1e-7:
                return Timer.pformat(interval, unit='us', **kwargs)
            else:
                return Timer.pformat(interval, unit='ns', **kwargs)
        elif unit == 's':
            if interval >= 60 and show_hms:
                hour = int(interval // 3600)
                minute = int((interval - hour * 3600) // 60)
                sec = int(interval - 3600*hour - 60*minute)
                if hour == 0:
                    hour = ''
                else:
                    hour = '{}h:'.format(hour)
                return '{}{:0>2d}m:{:0>2d}s'.format(hour, minute, sec)
            else:
                return (float_fmt + 's').format(interval)
        elif unit == 'ms':
            return (float_fmt + 'ms').format(interval * 1e3)
        elif unit == 'us':
            return (float_fmt + '\u03BCs').format(interval * 1e6)
        elif unit == 'ns':
            return (float_fmt + 'ns').format(interval * 1e9)
        else:
            raise ValueError('Unsupported unit: '+unit)

    def elapsed_str(self, unit='auto'):
        return self.pformat(self.elapsed(), unit=unit)

    def print_elapsed(self, msg='', unit='auto'):
        # in seconds:
        print(msg if msg else 'elapsed', self.elapsed_str(unit=unit))

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.print_elapsed()
