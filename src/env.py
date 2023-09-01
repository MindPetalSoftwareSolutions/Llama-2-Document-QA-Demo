from torch import cuda

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'