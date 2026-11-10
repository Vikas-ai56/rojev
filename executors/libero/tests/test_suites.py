"""Suite listing through LIBERO's benchmark registry (the fake one here) and LeRobot's step budgets."""
import pytest

import fake_libero
from suites import describe, load_suite, main, suite_names


def test_suites_are_listed_from_the_registry():
    assert suite_names() == sorted(fake_libero.SUITES)


def test_a_suite_lists_its_tasks_with_language_and_init_states():
    info = describe("libero_goal")
    assert info.max_steps == 300
    assert [t.index for t in info.tasks] == [0, 1, 2]
    assert [t.language for t in info.tasks] == list(fake_libero.SUITES["libero_goal"])
    assert {t.init_states for t in info.tasks} == {fake_libero.INIT_STATES}
    assert info.tasks[0].name == "open_the_middle_drawer_of_the_cabinet"


def test_an_unknown_suite_is_refused_with_the_registered_names():
    with pytest.raises(ValueError, match="libero_goal"):
        load_suite("libero_99")


def test_the_cli_prints_every_suite(capsys):
    assert main([]) == 0
    printed = capsys.readouterr().out
    assert "libero_goal: 3 tasks, max steps 300" in printed
    assert "libero_spatial: 1 tasks, max steps 280" in printed
