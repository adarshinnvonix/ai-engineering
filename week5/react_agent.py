"""Compatibility entry point for the agent example."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(Path(__file__).with_name("react-agent.py"), run_name="__main__")