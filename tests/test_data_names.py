from __future__ import annotations

from neuroseq.data_names import SIMULATION_GROUP
from neuroseq.data_names import data_name_problem
from neuroseq.data_names import group_name_problem
from neuroseq.data_names import split_data_name
from neuroseq.data_names import unique_group_name
from neuroseq.data_names import unique_name


def test_split_data_name() -> None:
    assert split_data_name("Rep 1 :: dF/F") == ("Rep 1", "dF/F")
    assert split_data_name("Rep 1") == ("Rep 1", "")


def test_data_name_problem() -> None:
    assert data_name_problem(" ok ") == ""
    assert data_name_problem("  ") == "The name cannot be empty."
    assert data_name_problem("a::b") == "The name cannot contain '::'."


def test_group_name_problem_reserves_simulation_and_existing_names() -> None:
    assert group_name_problem("New", ["Old"]) == ""
    assert "already exists" in group_name_problem(SIMULATION_GROUP, [])
    assert "already exists" in group_name_problem(" Old ", ["Old"])
    assert "cannot be empty" in group_name_problem("", [])


def test_unique_name_numbers_taken_names() -> None:
    assert unique_name("a", []) == "a"
    assert unique_name("a", ["a", "a (2)"]) == "a (3)"


def test_unique_group_name_is_always_a_valid_group_name() -> None:
    assert unique_group_name("Simulation", []) == "Simulation (2)"
    assert unique_group_name("a::b", []) == "a_b"
    assert unique_group_name("  ", []) == "Data"
    assert unique_group_name("rec", ["rec"]) == "rec (2)"
    for base in ["Simulation", "a :: b", "", "rec"]:
        name = unique_group_name(base, ["rec"])
        assert group_name_problem(name, ["rec"]) == ""
