
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class EngineConfig:
    def to_dict(self):
        """Return the configs as a dictionary, for use in **kwargs.
        """
        return dict(
            (field.name, getattr(self, field.name)) for field in fields(self))
