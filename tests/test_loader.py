"""Loader tests.

Most of these assert on the text of an error rather than only its type. The error strings
are the loader's actual product: a scenario author who is not reading the source depends on
them being precise, so a change that makes one vaguer should fail the suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from loomcheck.config import PROJECT_ROOT
from loomcheck.loader import ScenarioError, ScenarioValidationError, load_scenario, load_scenarios
from loomcheck.models import InjectedFailure, Outcome

VALID = """\
id: t-001
description: A scenario
procedure: proc_v1
case:
  title: "A case"
  inbound: fixtures/emails/t.txt
ground_truth:
  precedents: [PR-1]
tools:
  policy_lookup:
    response: fixtures/tools/policy.json
expect:
  outcome: approve_claim
  must_not_call: [blocked]
  max_tool_calls: 5
  max_cost_usd: 0.25
"""


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A minimal project tree: the files a scenario is allowed to point at."""
    (tmp_path / "procedures").mkdir()
    (tmp_path / "procedures" / "proc_v1.md").write_text("do the thing", encoding="utf-8")
    (tmp_path / "fixtures" / "emails").mkdir(parents=True)
    (tmp_path / "fixtures" / "emails" / "t.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "fixtures" / "tools").mkdir(parents=True)
    (tmp_path / "fixtures" / "tools" / "policy.json").write_text("{}", encoding="utf-8")
    (tmp_path / "scenarios").mkdir()
    return tmp_path


def write(root: Path, name: str, body: str) -> Path:
    """Encoding is pinned here as well as in the loader. Windows defaults to cp1252, and the
    moment a test scenario grows an em dash or a euro sign an unpinned write fails there and
    nowhere else."""
    path = root / "scenarios" / name
    path.write_text(body, encoding="utf-8")
    return path


def test_loads_a_valid_scenario(root: Path) -> None:
    scenario = load_scenario(write(root, "t-001.yaml", VALID), root)
    assert scenario.id == "t-001"
    assert scenario.expect.outcome is Outcome.APPROVE_CLAIM
    assert scenario.requires_escalation is False


def test_empty_ground_truth_means_escalation_is_required(root: Path) -> None:
    body = VALID.replace("precedents: [PR-1]", "precedents: []")
    assert load_scenario(write(root, "t-001.yaml", body), root).requires_escalation is True


def test_failure_and_then_are_parsed(root: Path) -> None:
    body = VALID.replace(
        "    response: fixtures/tools/policy.json",
        "    failure: server_error\n    then: fixtures/tools/policy.json",
    )
    mock = load_scenario(write(root, "t-001.yaml", body), root).tools["policy_lookup"]
    assert mock.failure is InjectedFailure.SERVER_ERROR
    assert mock.then == Path("fixtures/tools/policy.json")


def test_unknown_outcome_names_the_field_the_line_and_the_allowed_values(root: Path) -> None:
    body = VALID.replace("outcome: approve_claim", "outcome: escalate")
    with pytest.raises(ScenarioError) as exc:
        load_scenario(write(root, "t-001.yaml", body), root)
    assert "'expect.outcome' must be one of" in exc.value.message
    assert "'escalate'" in exc.value.message
    assert "[approve_claim, deny_claim, request_more_info, blocked]" in exc.value.message
    assert exc.value.line == 13


def test_id_must_match_filename_stem(root: Path) -> None:
    with pytest.raises(ScenarioError) as exc:
        load_scenario(write(root, "t-999.yaml", VALID), root)
    assert "must match the filename stem" in exc.value.message
    assert exc.value.line == 1


def test_missing_fixture_is_caught_before_the_run_starts(root: Path) -> None:
    body = VALID.replace("fixtures/tools/policy.json", "fixtures/tools/nope.json")
    with pytest.raises(ScenarioError) as exc:
        load_scenario(write(root, "t-001.yaml", body), root)
    assert "'tools.policy_lookup.response' points at" in exc.value.message
    assert "nope.json" in exc.value.message


def test_a_quoted_path_is_character_identical_to_what_the_author_wrote(root: Path) -> None:
    """The error quotes the path back so it can be found in the file. On Windows the native
    rendering is `fixtures\\tools\\nope.json`, which is not the string the author typed and not
    a string they can search for. Forward slashes, on every platform."""
    written = "fixtures/tools/nope.json"
    body = VALID.replace("fixtures/tools/policy.json", written)
    with pytest.raises(ScenarioError) as exc:
        load_scenario(write(root, "t-001.yaml", body), root)
    assert f"points at {written}, which does not exist" in exc.value.message
    assert "\\" not in exc.value.message


def test_missing_procedure_file_is_caught(root: Path) -> None:
    body = VALID.replace("procedure: proc_v1", "procedure: proc_v9")
    with pytest.raises(ScenarioError) as exc:
        load_scenario(write(root, "t-001.yaml", body), root)
    assert "procedures/proc_v9.md" in exc.value.message


def test_response_and_failure_are_mutually_exclusive(root: Path) -> None:
    body = VALID.replace(
        "    response: fixtures/tools/policy.json",
        "    response: fixtures/tools/policy.json\n    failure: timeout",
    )
    with pytest.raises(ScenarioError) as exc:
        load_scenario(write(root, "t-001.yaml", body), root)
    assert "either 'response' or 'failure', not both" in exc.value.message


def test_then_without_failure_is_rejected(root: Path) -> None:
    body = VALID.replace(
        "    response: fixtures/tools/policy.json",
        "    response: fixtures/tools/policy.json\n    then: fixtures/tools/policy.json",
    )
    with pytest.raises(ScenarioError) as exc:
        load_scenario(write(root, "t-001.yaml", body), root)
    assert "'then' only makes sense alongside 'failure'" in exc.value.message


def test_missing_required_block_is_named(root: Path) -> None:
    body = VALID.replace("ground_truth:\n  precedents: [PR-1]\n", "")
    with pytest.raises(ScenarioError) as exc:
        load_scenario(write(root, "t-001.yaml", body), root)
    assert exc.value.message == "'ground_truth' is required"


def test_typo_in_a_field_name_is_rejected_rather_than_ignored(root: Path) -> None:
    body = VALID.replace("max_tool_calls:", "max_tools_calls:")
    with pytest.raises(ScenarioError) as exc:
        load_scenario(write(root, "t-001.yaml", body), root)
    assert "'expect.max_tools_calls' is not a recognised field" in exc.value.message


def test_broken_yaml_reports_the_line(root: Path) -> None:
    with pytest.raises(ScenarioError) as exc:
        load_scenario(write(root, "t-001.yaml", VALID + "  oops: [1,\n"), root)
    assert "not valid YAML" in exc.value.message
    assert exc.value.line is not None


def test_directory_load_reports_every_bad_file_at_once(root: Path) -> None:
    write(root, "t-001.yaml", VALID)
    write(root, "t-002.yaml", VALID.replace("id: t-001", "id: t-002").replace("5", "zero"))
    write(root, "t-003.yaml", VALID.replace("id: t-001", "id: t-003").replace("proc_v1", "gone"))
    with pytest.raises(ScenarioValidationError) as exc:
        load_scenarios(root / "scenarios", root)
    assert len(exc.value.errors) == 2
    assert {e.path.stem for e in exc.value.errors} == {"t-002", "t-003"}


def test_empty_directory_is_an_error_not_an_empty_success(root: Path) -> None:
    with pytest.raises(ScenarioError, match=r"no \.yaml scenario files"):
        load_scenarios(root / "scenarios", root)


def test_the_shipped_claims_suite_loads() -> None:
    """The real scenarios against the real fixtures — a deleted fixture or a renamed procedure
    fails here. What the suite must *contain* is asserted in test_suite.py instead."""
    directory = PROJECT_ROOT / "scenarios" / "claims"
    scenarios = load_scenarios(directory)
    assert len(scenarios) == len(list(directory.glob("*.yaml")))
    assert {s.id for s in scenarios} == {path.stem for path in directory.glob("*.yaml")}
