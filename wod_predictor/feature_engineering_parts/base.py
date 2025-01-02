from abc import ABC, abstractmethod
from contextlib import contextmanager


class FitContext:
    """
    Manages a global fit state using a context manager.
    """

    fit_mode = False

    @classmethod
    def is_fitting(cls):
        return cls.fit_mode

    @classmethod
    @contextmanager
    def fit_mode_on(cls):
        """
        Activate fit mode for the duration of the context.
        """
        cls.fit_mode = True
        try:
            yield
        finally:
            cls.fit_mode = False


class TransformerMixIn(ABC):
    def __init__(self):
        self.is_fit = False

    def fit(self, *args, **kwargs):
        self.is_fit = True

    def fit_transform(self, *args, **kwargs):
        """
        Combines fit and transform steps.
        Subclasses can optionally override this for more specific behavior.
        """
        with FitContext.fit_mode_on():
            transformed_data = self.transform(*args, **kwargs)
        return transformed_data

    def check_fit(self, *args, **kwargs):
        if FitContext.fit_mode:
            self.fit(*args, **kwargs)
        if not self.is_fit:
            raise ValueError("Fit has not been called yet")

    @abstractmethod
    def transform(self, *args, **kwargs):
        """
        Transform the data. If `fit=True`, this method should also perform fitting.
        Must be implemented by subclasses.
        """
        pass
