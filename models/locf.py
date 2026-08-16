"""Last Observation Carried Forward model adapter."""

from .base import AbstractModel


class LOCFModel(AbstractModel):
    name = "locf"
    requires_training = False

    def _build_backend(self):
        from pypots.imputation import LOCF

        return LOCF(first_step_imputation="zero")
