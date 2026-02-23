"""
Pytest configuration and fixtures for enhanced error reporting.

This module provides:
1. Custom test markers for better organization
2. Enhanced logging for test failures
3. Hooks for better error reporting
"""
import sys

import pytest


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Make test results available to fixtures for cleanup decisions.

    This hook allows fixtures to check if a test failed and adjust cleanup accordingly.
    Access via: item.rep_setup, item.rep_call, item.rep_teardown
    """
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


def pytest_runtest_logreport(report):
    """
    Enhanced logging for test failures.

    This hook is called when a test generates a report (setup, call, or teardown).
    We use it to print detailed failure information to stderr.
    """
    if report.failed:
        print(f"\n{'=' * 70}", file=sys.stderr)
        print(f"TEST FAILED: {report.nodeid}", file=sys.stderr)
        print(f"Phase: {report.when}", file=sys.stderr)
        print(f"{'=' * 70}", file=sys.stderr)

        if report.longrepr:
            print(report.longreprtext, file=sys.stderr)

        print(f"{'=' * 70}\n", file=sys.stderr)


def pytest_exception_interact(node, call, report):
    """
    Called when an exception is raised during test execution.

    This provides an additional hook to capture and report exceptions,
    especially useful for debugging threading issues.
    """
    if call.excinfo is not None:
        print(f"\n{'=' * 70}", file=sys.stderr)
        print(f"EXCEPTION DURING TEST: {node.nodeid}", file=sys.stderr)
        print(f"Exception Type: {call.excinfo.typename}", file=sys.stderr)
        print(f"Exception Value: {call.excinfo.value}", file=sys.stderr)
        print(f"{'=' * 70}", file=sys.stderr)
        print(call.excinfo.getrepr(style="long"), file=sys.stderr)
        print(f"{'=' * 70}\n", file=sys.stderr)


@pytest.fixture(scope="session", autouse=True)
def session_setup():
    """
    Session-level fixture that runs once before all tests.

    Use this for one-time setup and to print helpful debug information.
    """
    print("\n" + "=" * 70)
    print("TEST SESSION STARTING")
    print("=" * 70)
    print(f"Python: {sys.version}")
    print(f"Pytest: {pytest.__version__}")
    print("=" * 70 + "\n")

    yield

    print("\n" + "=" * 70)
    print("TEST SESSION COMPLETE")
    print("=" * 70 + "\n")
