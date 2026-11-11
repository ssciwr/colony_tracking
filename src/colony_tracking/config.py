"""Shared config machinery: the config error type, value checking, and the
flat-document reader/writer every run type uses.

A *flat* config document is one TOML table whose keys are exactly the field
names of a run type's :class:`~colony_tracking._options.Options` dataclass —
what the TUI writes for a stage or finetune step, and what the CLI's
``--config`` accepts for one. :class:`FlatConfig` owns that format in one
place: structural validation (unknown keys), full resolution (missing
required fields and mistyped values), TOML reading, and writing.

The error wordings are the pipeline config's: unknown keys
(``unknown key(s) [...] — expected one key per name: [...]``), missing
required keys (``missing required key(s): [...]``), and mistyped values
(``<run type>: key 'x' must be <type> (got <value>)``). Every error is
logged through :data:`colony_tracking.log` under its owner's prefix and
raised as :class:`ConfigError`, which the CLI maps to exit code 2.
"""

from __future__ import annotations

import tomllib
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Literal, get_args, get_origin, get_type_hints

import tomli_w

from . import log
from ._options import Options, _from_plain, _union_members


class ConfigError(ValueError):
    """An invalid config: unknown key/section, missing required key, mistyped
    value, or malformed TOML.

    Logged with a clear message and raised; the CLI maps it to exit code 2.
    """


def config_error(prefix: str, message: str) -> ConfigError:
    """Log a config error under its owner's prefix, then return it for
    raising (``pipeline config: ...``, ``train config: ...``)."""
    log.error("%s config: %s", prefix, message)
    return ConfigError(message)


#: Scalar hint → (plain-value predicate, phrase, plural phrase): one table
#: shared by ``_human_hint`` and ``_compatible`` so the wording cannot drift.
_SCALARS: dict[type, tuple[Callable[[Any], bool], str, str]] = {
    Path: (lambda v: isinstance(v, str), "a path string", "path strings"),
    bool: (lambda v: isinstance(v, bool), "true or false", "booleans"),
    int: (lambda v: isinstance(v, int) and not isinstance(v, bool),
          "an integer", "integers"),
    float: (lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "a number", "numbers"),
    str: (lambda v: isinstance(v, str), "a string", "strings"),
}
#: phrase → plural, derived from the same table.
_PLURAL = {phrase: plural for _, (_, phrase, plural) in _SCALARS.items()}


def _human_hint(hint: Any) -> str:
    """A human-readable description of a type hint (for error messages)."""
    origin, args = get_origin(hint), get_args(hint)
    if origin is Literal:
        return " or ".join(repr(a) for a in args)
    if origin in (list, tuple):
        item = _human_hint(args[0])
        return f"a list of {_PLURAL.get(item, item)}"
    if (members := _union_members(hint)) is not None:  # a Union
        return " or ".join(_human_hint(a) for a in members)
    if args:  # another parameterized hint: list its members
        return " or ".join(_human_hint(a) for a in args if a is not type(None))
    if hint in _SCALARS:
        return _SCALARS[hint][1]
    return getattr(hint, "__name__", str(hint))


def _compatible(value: Any, hint: Any) -> bool:
    """Whether a plain (TOML) value can be read as the declared field type."""
    origin, args = get_origin(hint), get_args(hint)
    if origin is Literal:
        return value in args
    if origin in (list, tuple):
        return isinstance(value, list) and all(
            _compatible(v, args[0]) for v in value)
    if (members := _union_members(hint)) is not None:  # a Union
        return (value is None and type(None) in args) or any(
            _compatible(value, a) for a in members)
    scalar = _SCALARS.get(hint)
    return scalar[0](value) if scalar else True  # unhinted/unknown: accept


def value_message(where: str, name: str, value: Any, hint: Any) -> str:
    """The one mistyped-value wording (``<where>: key 'x' must be ...``)."""
    return f"{where}: key {name!r} must be {_human_hint(hint)} (got {value!r})"


@dataclass(frozen=True)
class FlatConfig:
    """One run type's flat config document (one key per option field).

    ``run_type`` is the document's owner (``train``, ``prepare``, ...): it
    prefixes the logged errors and the mistyped-value message. ``options``
    is the :class:`~colony_tracking._options.Options` dataclass the document
    describes.
    """

    run_type: str
    options: type[Options]

    @property
    def names(self) -> list[str]:
        """The document's field names, in dataclass declaration order."""
        return [f.name for f in fields(self.options)]

    def validate_structure(self, doc: Any) -> dict[str, Any]:
        """Structural validation only: the document must be a table whose
        keys are exactly the dataclass's field names.

        Unknown keys refuse; missing requireds and mistyped values pass
        (a load reports them at the next :meth:`resolve`). Returns ``doc``.
        """
        names = self.names
        if not isinstance(doc, dict):
            raise config_error(
                self.run_type,
                f"{doc!r} is not a table of {self.run_type} options",
            )
        unknown = sorted(set(doc) - set(names))
        if unknown:
            raise config_error(
                self.run_type,
                f"unknown key(s) {unknown} — expected one key per name: "
                f"{sorted(names)}",
            )
        return doc

    def resolve(self, doc: Any) -> Options:
        """Validate and build the options from a flat document.

        Checks, in order: the document is a table; its keys are exactly the
        field names; every field without a default is present; every value
        matches its declared type. Omitted keys take their dataclass
        defaults, and values coerce through
        :func:`~colony_tracking._options._from_plain` exactly as a read-back
        TOML value would.
        """
        self.validate_structure(doc)
        declared = fields(self.options)
        missing = [
            f.name
            for f in declared
            if f.name not in doc
            and f.default is MISSING
            and f.default_factory is MISSING
        ]
        if missing:
            raise config_error(self.run_type, f"missing required key(s): {missing}")
        hints = get_type_hints(self.options)
        for key, value in doc.items():
            hint = hints[key]
            if not _compatible(value, hint):
                raise config_error(
                    self.run_type,
                    value_message(self.run_type, key, value, hint),
                )
        kwargs = {key: _from_plain(value, hints[key]) for key, value in doc.items()}
        return self.options(**kwargs)

    def document(self, options: Options) -> dict[str, Any]:
        """The effective document of resolved ``options``: every field,
        defaults included, with unset (``None``) values omitted."""
        return {
            key: value
            for key, value in options.to_dict().items()
            if value is not None
        }

    def read_document(self, path: Path | str) -> dict[str, Any]:
        """Read a flat TOML document (no structural validation)."""
        path = Path(path)
        try:
            with open(path, "rb") as handle:
                doc = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise config_error(
                self.run_type, f"{path}: not valid TOML: {exc}"
            ) from exc
        log.info("%s config: reading %s", self.run_type, path)
        return doc

    def read(self, path: Path | str) -> Options:
        """Read and resolve a flat TOML document."""
        return self.resolve(self.read_document(path))

    def write(self, options: Options, path: Path | str) -> Path:
        """Write ``options`` as a self-contained flat TOML document.

        The effective document (:meth:`document` — defaults included, unset
        values omitted) is written whole. Parent directories are created.
        Returns the path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            tomli_w.dump(self.document(options), handle)
        log.info("%s config: written to %s", self.run_type, path)
        return path
