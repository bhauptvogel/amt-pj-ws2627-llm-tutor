"""A named lookup of implementations of one seam, filled by the implementations."""


class RegistryError(Exception):
    """A registry contract violation: a duplicate or an unknown name."""


class Registry[T]:
    """Names to classes, with duplicate detection and a helpful unknown-name error."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: dict[str, type[T]] = {}

    def register(self, name: str, implementation: type[T]) -> None:
        if name in self._entries:
            raise RegistryError(f"duplicate {self._kind} name {name!r}")
        self._entries[name] = implementation

    def get(self, name: str) -> type[T]:
        try:
            return self._entries[name]
        except KeyError:
            registered = ", ".join(sorted(self._entries)) or "(none)"
            raise RegistryError(
                f"unknown {self._kind} {name!r}; registered: {registered}"
            ) from None

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))
