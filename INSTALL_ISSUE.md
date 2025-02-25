#### 1. Cannot import zero_gradients from torch.autograd.gradcheck

- modify an import in `site-packages/advertorch/attacks/fast_adaptive_boundary.py`
    ```python
    try:
        from torch.autograd.gradcheck import zero_gradients
    except ImportError:
        def zero_gradients(x):
            if isinstance(x, torch.Tensor):
                if x.grad is not None:
                    x.grad.detach_()
                    x.grad.zero_()
            elif isinstance(x, collections.abc.Iterable):
                for elem in x:
                    zero_gradients(elem)
    ```




