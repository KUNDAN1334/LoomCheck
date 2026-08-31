"""YAML on disk to validated Scenario objects.

The whole value of this module is its error messages. A scenario suite is written by
someone who is not reading the source, so a failure has to say which file, which line,
which field, and what the acceptable values are. Anything less and the author guesses.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError
from pydantic_core import ErrorDetails

from loomcheck.config import PROJECT_ROOT
from loomcheck.models import Scenario


class ScenarioError(Exception):
    """One problem with one scenario file, addressed well enough to fix without guessing."""

    def __init__(self, path: Path, message: str, line: int | None = None) -> None:
        self.path = path
        self.message = message
        self.line = line
        location = f"{path}:{line}" if line is not None else str(path)
        super().__init__(f"{location}: {message}")


class ScenarioValidationError(Exception):
    """Every problem found across a directory, so one pass fixes all of them."""

    def __init__(self, errors: list[ScenarioError]) -> None:
        self.errors = errors
        super().__init__(f"{len(errors)} scenario file(s) failed validation")


def _line_of(root: yaml.Node | None, loc: tuple[int | str, ...]) -> int | None:
    """Find the source line of a field, given the Pydantic error location path.

    Walks the composed YAML node tree rather than the parsed dict, because only the node
    tree keeps source marks. Falls back to the deepest node it could reach, so an error on
    a missing key still points at the block the key belongs in.
    """
    if root is None:
        return None
    node: yaml.Node = root
    for key in loc:
        if isinstance(node, yaml.MappingNode) and isinstance(key, str):
            match = next((v for k, v in node.value if k.value == key), None)
            if match is None:
                break
            node = match
        elif isinstance(node, yaml.SequenceNode) and isinstance(key, int) and key < len(node.value):
            node = node.value[key]
        else:
            break
    return int(node.start_mark.line) + 1


def _primary(errors: list[ErrorDetails]) -> ErrorDetails:
    """Choose the error worth showing when Pydantic reports several for one file.

    A misspelled key raises two: the real key is 'missing' and the typo is 'extra_forbidden'.
    Reporting the missing key sends the author looking for a field they thought they wrote,
    so the typo wins.
    """
    return next((e for e in errors if e["type"] == "extra_forbidden"), errors[0])


def _describe(error: ErrorDetails) -> str:
    """Turn one Pydantic error into a sentence that names the fix."""
    field = ".".join(str(part) for part in error["loc"]) or "<document>"
    kind = error["type"]
    ctx = error.get("ctx", {})
    if kind == "enum":
        allowed = str(ctx["expected"]).replace(" or ", ", ").replace("'", "")
        return f"'{field}' must be one of [{allowed}], got {error['input']!r}"
    if kind == "missing":
        return f"'{field}' is required"
    if kind == "extra_forbidden":
        return f"'{field}' is not a recognised field"
    if kind == "value_error":
        return f"'{field}': {str(ctx.get('error', error['msg'])).removeprefix('Value error, ')}"
    return f"'{field}': {error['msg']}"


def load_scenario(path: Path, root: Path = PROJECT_ROOT) -> Scenario:
    """Load and fully validate one scenario file.

    `root` is what the scenario's fixture paths are relative to. It is a parameter rather
    than a constant so tests can build a scenario tree in a tmpdir without touching the repo.
    """
    text = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
        nodes = yaml.compose(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        line = mark.line + 1 if mark is not None else None
        raise ScenarioError(path, f"not valid YAML: {getattr(exc, 'problem', exc)}", line) from exc

    if not isinstance(data, dict):
        raise ScenarioError(path, "expected a mapping at the top level")

    try:
        scenario = Scenario.model_validate(data)
    except ValidationError as exc:
        problem = _primary(exc.errors())
        raise ScenarioError(path, _describe(problem), _line_of(nodes, problem["loc"])) from exc

    _check_id_matches_filename(scenario, path, nodes)
    _check_referenced_files_exist(scenario, path, root, nodes)
    return scenario


def _check_id_matches_filename(scenario: Scenario, path: Path, nodes: yaml.Node | None) -> None:
    """Filenames are unique within a directory, so tying id to filename makes ids unique too.

    Without this, two scenarios could share an id and silently overwrite each other's result
    in a diff, which would look like a passing suite.
    """
    if scenario.id != path.stem:
        raise ScenarioError(
            path,
            f"'id' must match the filename stem: expected {path.stem!r}, got {scenario.id!r}",
            _line_of(nodes, ("id",)),
        )


def procedure_path(name: str, root: Path = PROJECT_ROOT) -> Path:
    """Locate a procedure file, failing with a message that lists the ones that exist.

    Used by the loader and by `--procedure`, so a typo on the command line is caught before a
    suite starts rather than on the first scenario.
    """
    path = root / "procedures" / f"{name}.md"
    if not path.is_file():
        available = ", ".join(sorted(p.stem for p in (root / "procedures").glob("*.md"))) or "none"
        raise ScenarioError(path, f"no such procedure {name!r} (available: {available})")
    return path


def _check_referenced_files_exist(
    scenario: Scenario, path: Path, root: Path, nodes: yaml.Node | None
) -> None:
    """Catch a missing fixture now, not three minutes into a paid suite run.

    Paths quoted back to the author are rendered with forward slashes, whatever the platform,
    because that is how they are written in the YAML being reported on. On Windows the native
    form would be `fixtures\\emails\\wd_004.txt`, and an author looking for that string in their
    own file would not find it. The file location in the error prefix stays native — that one is
    a real filesystem path someone pastes into an editor.
    """
    referenced: list[tuple[tuple[int | str, ...], Path]] = [
        (("procedure",), Path("procedures") / f"{scenario.procedure}.md"),
        (("case", "inbound"), scenario.case.inbound),
    ]
    for tool_name, mock in scenario.tools.items():
        for fixture in mock.fixture_paths():
            field = "then" if fixture == mock.then else "response"
            referenced.append((("tools", tool_name, field), fixture))

    for loc, relative in referenced:
        if not (root / relative).is_file():
            field = ".".join(str(part) for part in loc)
            raise ScenarioError(
                path,
                f"'{field}' points at {relative.as_posix()}, which does not exist",
                _line_of(nodes, loc),
            )


def load_scenarios(target: Path, root: Path = PROJECT_ROOT) -> list[Scenario]:
    """Load one file or every .yaml in a directory, reporting all bad files at once."""
    if target.is_dir():
        paths = sorted(target.glob("*.yaml"))
        if not paths:
            raise ScenarioError(target, "directory contains no .yaml scenario files")
    elif target.is_file():
        paths = [target]
    else:
        raise ScenarioError(target, "no such file or directory")

    scenarios: list[Scenario] = []
    errors: list[ScenarioError] = []
    for path in paths:
        try:
            scenarios.append(load_scenario(path, root))
        except ScenarioError as exc:
            errors.append(exc)

    if errors:
        raise ScenarioValidationError(errors)
    _check_single_procedure(scenarios, paths)
    return scenarios


def _check_single_procedure(scenarios: list[Scenario], paths: list[Path]) -> None:
    """A run compares one procedure against another, so a run executes exactly one.

    `runs.procedure` is a single column and the diff is built to attribute a change to a
    procedure edit. A directory mixing two would produce a run whose recorded procedure is
    true of only some of its rows.
    """
    if not scenarios:
        return
    expected = scenarios[0].procedure
    for scenario, path in zip(scenarios, paths, strict=True):
        if scenario.procedure != expected:
            raise ScenarioError(
                path,
                f"'procedure' is {scenario.procedure!r}, but {paths[0].name} in the same run "
                f"uses {expected!r}; one run executes one procedure",
            )
