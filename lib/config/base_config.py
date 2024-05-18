import os
from utils import pr

class BaseConfig():

    def print_to_screen(self, offset=0):
        lines = []
        term_width = os.get_terminal_size().columns - offset
        left_width = min(max([len(k) for k in vars(self).keys()]) + 2, term_width // 3)
        right_width = term_width - left_width
        lines.append('=' * term_width)
        lines.append(f'[{self.__class__.__name__}]')
        lines.append('-' * term_width)
        if hasattr(self, "comments"):
            if len(self.comments) > right_width:
                comments = self.comments[:right_width // 2] + '...' + self.comments[-right_width + 4:]
            else:
                comments = self.comments
            lines.append(f'{"comments":<{left_width}} {comments}')
        for k, v in vars(self).items():
            if k in ['data_root', 'network', 'trainer', 'save_root', 'decom_params']:
                lines.append('-' * term_width)
            if k not in ['comments', 'other_info']:
                if len(k) > left_width:
                    k = k[:left_width // 2 - 4] + '...' + k[-left_width // 2:]
                if len(str(v)) > right_width:
                    v = str(v)[:right_width // 2] + '...' + str(v)[-right_width // 2 + 4:]
                lines.append(f'{k:<{left_width}} {v}')
        lines.append('=' * term_width)
        pr('\n'.join(lines))
        return lines

    def to_lines(self):
        lines = []
        lines.append('=' * 80)
        lines.append(f'[{self.__class__.__name__}]')
        lines.append('-' * 80)
        if hasattr(self, "comments"):
            lines.append(f'{"comments":<30} {self.comments}')
        for k, v in vars(self).items():
            if k in ['data_root', 'network', 'trainer', 'save_root', 'decoder_params']:
                lines.append('')
            if k not in ['comments', 'other_info']:
                lines.append(f'{k:<30} {v}')
        lines.append('=' * 80)
        return lines

    def __repr__(self) -> str:
        string = ''
        for line in self.to_lines():
            string += line + '\n'
        return string