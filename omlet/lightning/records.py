import os
import omlet.utils as U
from pytorch_lightning.callbacks import Callback


class SourceCodeBackup(Callback):
    master_only = True

    def __init__(self,
                 exp_dir,
                 source_dir,
                 include_pattern=('*.py', '*.sh'),
                 subdir='code'):
        """
        Args:
            tag_var_map: maps a tensorboard tag to the tracked variable
        """
        self.record_dir = os.path.join(os.path.expanduser(exp_dir), subdir)
        self.source_dir = self._get_code_path(source_dir)
        assert os.path.exists(self.source_dir), \
            'source code dir "{}" does not exist'.format(self.source_dir)
        self.include_pattern = include_pattern

    def _get_code_path(self, path):
        "handles both abspath and relative path"
        path = os.path.expanduser(path)
        if os.path.isabs(path):
            return path
        else:
            # relative to the current file
            current_script = eval('__file__')
            path = os.path.join(os.path.dirname(current_script), path)
            return os.path.abspath(path)

    def on_init_start(self, trainer):
        U.f_mkdir(self.record_dir)
        U.f_copytree(
            self.source_dir, self.record_dir,
            include=self.include_pattern, exist_ok=True
        )
        print(
            'Backed up source code {} to {}', self.source_dir, self.record_dir
        )

