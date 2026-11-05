"""Timeout for qtbot waits, set from pytest's ``--wait-timeout`` option.

The waits return as soon as their condition is met, so the timeout only
matters when a test hangs (short by default, for quick feedback locally) or a
run is slow (CI passes a longer one). conftest.py sets it before the test
modules are collected, so they can import it by value.
"""

WAIT_TIMEOUT_MS = 5000
