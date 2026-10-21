"""ConfigDraft: the TUI's single source of truth for a pipeline config.

The TUI never edits a partially-resolved
:class:`~colony_tracking.pipeline.PipelineConfig`; it edits one plain,
TOML-shaped document — the ``[pipeline]`` run-level section plus one
optional section per stage, holding plain values only (paths as strings,
tuples as lists) so a TOML round-trip is exact.

The pipeline-owned fields (:data:`colony_tracking.pipeline._OWNED` —
``[difference].raw_dir``/``reference``, ``[segmentation].input``/
``diff_dir``, ``[analysis].raw_dir``) are never stored in a draft: they are
derived from the run-level settings and the chain wiring, the stage screens
show them as derived/read-only, and ``write_config`` omits them from the
written file anyway. ``set`` refuses them with the pipeline's own reason.

Validation happens in :meth:`ConfigDraft.resolve` only: it goes through
``PipelineConfig.from_dict``, so every invalid draft raises the pipeline's
own :class:`~colony_tracking.pipeline.ConfigError` with its exact message —
the same error the CLI reports for the equivalent TOML file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .._options import _to_plain
from ..pipeline import (
    PipelineConfig,
    _OWNED,
    _OWNED_REASON,
    _STAGES,
    _config_error,
    write_config,
)

#: The run-level section of a draft document (``[pipeline]``).
RUN_LEVEL = "pipeline"

#: The stage sections of a draft document (the config's stage names).
STAGES: tuple[str, ...] = _STAGES


def _copy_value(value: Any) -> Any:
    """A copy of one plain draft value (a list value must not alias the
    draft's own, or editing a document copy could corrupt the draft)."""
    return list(value) if isinstance(value, list) else value


class ConfigDraft:
    """The in-progress pipeline config document (see the module docstring).

    The draft is the TUI's only config state: every screen edits it, and
    every action (write the TOML, run a stage, run the whole pipeline)
    goes through :meth:`resolve` / :meth:`write`. A key that is not set
    takes the stage's default at ``resolve`` time, exactly like an omitted
    key in a TOML config file.
    """

    def __init__(self) -> None:
        #: The run-level (``[pipeline]``) section: plain values only.
        self.run_level: dict[str, Any] = {}
        #: The stage sections: stage name -> plain values (may be empty).
        self.stages: dict[str, dict[str, Any]] = {stage: {} for stage in STAGES}

    def _section(self, stage: str) -> dict[str, Any]:
        """The named section: the run-level one for ``pipeline``, else the
        stage's own (a :class:`ConfigError` for an unknown section)."""
        if stage == RUN_LEVEL:
            return self.run_level
        if stage in self.stages:
            return self.stages[stage]
        raise _config_error(
            f"unknown draft section {stage!r} — expected [{RUN_LEVEL}] or "
            f"one of the stage sections {list(STAGES)}"
        )

    def set(self, stage: str, key: str, value: Any) -> None:
        """Set ``key`` in section ``stage`` (``pipeline`` for a run-level
        setting) to the plain value ``value``.

        ``Path`` values are stored as strings and tuples as lists, so the
        document stays TOML-shaped and round-trips exactly. An owned key
        (:data:`colony_tracking.pipeline._OWNED`) or an unknown section is
        refused with :class:`~colony_tracking.pipeline.ConfigError`.
        """
        section = self._section(stage)
        # A section added after construction (a future stage's) owns
        # nothing yet; _OWNED only knows the pipeline's own stages.
        if stage != RUN_LEVEL and key in _OWNED.get(stage, frozenset()):
            raise _config_error(
                f"[{stage}] key {key!r} is owned by the pipeline — "
                f"{_OWNED_REASON[stage, key]}"
            )
        section[key] = _to_plain(value)

    def get(self, stage: str, key: str, default: Any = None) -> Any:
        """The value of ``key`` in section ``stage``, or ``default`` when
        the key is unset (the stage default then applies at ``resolve``)."""
        return self._section(stage).get(key, default)

    def clear(self, stage: str, key: str) -> None:
        """Drop ``key`` from section ``stage`` — back to the stage default
        (dropping an unset key is a no-op)."""
        self._section(stage).pop(key, None)

    def document(self) -> dict[str, dict[str, Any]]:
        """A TOML-shaped copy of the draft: the ``[pipeline]`` section plus
        the non-empty stage sections.

        The copy is detached from the draft (editing it cannot corrupt the
        draft), which makes it safe to hand to config resolution, TOML
        preview, and tests alike.
        """
        doc: dict[str, dict[str, Any]] = {
            RUN_LEVEL: {
                key: _copy_value(value) for key, value in self.run_level.items()
            }
        }
        for stage, section in self.stages.items():
            if section:
                doc[stage] = {key: _copy_value(value) for key, value in section.items()}
        return doc

    def resolve(self) -> PipelineConfig:
        """The resolved config for the draft as it stands right now.

        Goes through ``PipelineConfig.from_dict``: owned fields are derived
        from the run level and the chain wiring, defaults fill every unset
        key, and every invalid draft raises
        :class:`~colony_tracking.pipeline.ConfigError` with the pipeline's
        own message.
        """
        return PipelineConfig.from_dict(self.document())

    def write(self, path: Path | str) -> PipelineConfig:
        """Resolve the draft and write it as a TOML config file (a run
        started from the draft writes the same file first, so the run is
        reproducible). Returns the resolved config."""
        config = self.resolve()
        write_config(config, path)
        return config
