"""README contract tests: the option tables must match the live CLI.

The README's per-subcommand option tables are generated from the same source
the CLI parses — the argparse actions built by :mod:`colony_tracking.cli` —
so they cannot drift from the option dataclasses: every flag, its default,
and its help text must appear verbatim (as one rendered table) in the
README's section for that subcommand. When a dataclass, a flag rename, or a
help text changes, regenerate the README tables by running this module as a
script and pasting its output:

    uv run python tests/test_readme.py

Rendering rules (the Default column, backticked): ``required`` for flags
without a default, ``none`` for ``None``, ``unset`` for booleans (the flag
is absent unless given; the help text states the value in force),
``512 512`` for n-value tuples, otherwise the plain value.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from colony_tracking import cli

#: The README at the repo root.
README_PATH = Path(__file__).resolve().parent.parent / "README.md"

#: Subcommand path (as nested under ``train``) → its README section heading.
#: A section spans from its heading to the next heading of the same or a
#: higher level, so deeper subsections stay inside it.
SECTION_HEADING = {
    ("difference",): "### difference",
    ("segment",): "### segment",
    ("analyze",): "### analyze",
    ("pipeline",): "### pipeline",
    ("train",): "### train",
    ("train", "prepare"): "#### prepare",
    ("train", "train"): "#### train",
    ("train", "evaluate"): "#### evaluate",
}


def _subcommands(parser: argparse.ArgumentParser) -> dict[tuple[str, ...],
                                                          argparse.ArgumentParser]:
    """Every subcommand as ``path → subparser``, depth-first, in definition
    order."""
    found: dict[tuple[str, ...], argparse.ArgumentParser] = {}
    for action in parser._actions:  # noqa: SLF001
        if not isinstance(action, argparse._SubParsersAction):
            continue
        for name, sub in action.choices.items():
            found[(name,)] = sub
            for path, nested in _subcommands(sub).items():
                found[(name, *path)] = nested
    return found


#: Every subcommand's parser, built once (the CLI builds no heavy imports:
#: micro-sam/torch are imported lazily inside the segment step only).
SUBCOMMANDS = _subcommands(cli._build_parser())


def _default_cell(action: argparse.Action) -> str:
    """The Default column for one flag (see the module docstring)."""
    if action.required:
        return "`required`"
    default = action.default
    if isinstance(default, bool):
        return "`unset`"
    if default is None:
        return "`none`"
    if isinstance(default, tuple):
        return "`" + " ".join(str(v) for v in default) + "`"
    return f"`{default}`"


def _help_cell(action: argparse.Action) -> str:
    """The help text, with table-hostile pipes escaped."""
    return (action.help or "").replace("|", "\\|")


def _render_table(parser: argparse.ArgumentParser) -> str:
    """The README option table for one subcommand: one row per flag, in
    definition (dataclass field) order."""
    rows = [
        f"| `{action.option_strings[0]}` | {_default_cell(action)} | "
        f"{_help_cell(action)} |"
        for action in parser._actions  # noqa: SLF001
        if action.option_strings and "-h" not in action.option_strings
    ]
    if not rows:
        return ""
    return "\n".join([
        "| Option | Default | Description |",
        "| --- | --- | --- |",
        *rows,
    ])


def _section(readme: str, heading: str) -> str:
    """The README section under ``heading`` (to the next same-or-higher
    heading), with the heading line itself kept."""
    lines = readme.splitlines()
    level = len(heading) - len(heading.lstrip("#"))
    for start, line in enumerate(lines):
        if line.strip() == heading:
            break
    else:
        raise AssertionError(f"README has no {heading!r} section")
    kept = [lines[start]]
    for line in lines[start + 1:]:
        stripped = line.lstrip()
        if stripped.startswith("#"):
            next_level = len(stripped) - len(stripped.lstrip("#"))
            if next_level <= level:
                break
        kept.append(line)
    return "\n".join(kept)


def _table_paths() -> list[tuple[str, ...]]:
    """Subcommand paths that carry flags (the ``train`` parent has none)."""
    return sorted(
        path for path, parser in SUBCOMMANDS.items() if _render_table(parser)
    )


_TABLE_PATHS = _table_paths()


@pytest.mark.parametrize("path", _TABLE_PATHS,
                         ids=["-".join(p) for p in _TABLE_PATHS])
def test_option_table_matches_the_cli(path: tuple[str, ...]):
    """The README section carries the CLI's exact option table."""
    readme = README_PATH.read_text(encoding="utf-8")
    table = _render_table(SUBCOMMANDS[path])
    section = _section(readme, SECTION_HEADING[path])
    assert table in section, (
        f"README section {SECTION_HEADING[path]!r} does not carry the CLI's "
        f"option table (regenerate: uv run python tests/test_readme.py)\n"
        f"--- expected table ---\n{table}"
    )


def test_every_subcommand_is_documented():
    """Each subcommand — and the train parent — has its README heading."""
    readme = README_PATH.read_text(encoding="utf-8")
    for path, heading in SECTION_HEADING.items():
        assert heading in readme, f"README is missing the {heading!r} heading"


def test_readme_documents_the_integration_test():
    """The marked integration test's env var, GPU need, and invocation."""
    readme = README_PATH.read_text(encoding="utf-8")
    for needle in ("COLONY_TRACKING_DATA_ROOT", "uv run pytest -m integration",
                   "GPU"):
        assert needle in readme, f"README does not document {needle!r}"


def test_readme_documents_the_layouts_and_outputs():
    """Both input layouts and the output conventions are documented."""
    readme = README_PATH.read_text(encoding="utf-8")
    for needle in ("flat", "per-well", "colonies.csv", "_done.txt",
                   "scale.txt", "well_geometry.csv", "__labels.tif",
                   "ancestors", "GLOSSARY.md", "docs/adr/"):
        assert needle in readme, f"README does not document {needle!r}"


if __name__ == "__main__":  # regenerate the README tables (paste the output)
    for path, parser in SUBCOMMANDS.items():
        table = _render_table(parser)
        if not table:
            continue
        print(f"\n<!-- {SECTION_HEADING[path]} -->\n{table}\n")