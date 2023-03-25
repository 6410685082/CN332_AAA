from .scheduler_adapter import CeleryAdapter
app = CeleryAdapter.app
__all__ = ['app']
