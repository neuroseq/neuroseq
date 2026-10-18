from collections.abc import Callable
from functools import wraps

from neuroseq.project import Project

ExampleProjectGenerator = Callable[[], Project]

example_projects: dict[str, dict[str, ExampleProjectGenerator]] = {}


def register_example_project(name: str, section: str):
    """
    Decorator used to register example project generator functions so that they will be displayed in the GUI

    Args:
        name: The name shown in the GUI (e.g., "Non-linear").
        section: Submenu section in "Open example project".
    """

    def decorator(func: ExampleProjectGenerator) -> ExampleProjectGenerator:
        @wraps(func)
        def wrapped() -> Project:
            project = func()
            project.project_name = name
            project.project_section = section
            # Built up via Project's own mutating API (add_stim_array, etc.),
            # which marks dirty as a side effect - but nobody has actually
            # edited anything yet, so this shouldn't trigger a save prompt
            # the moment it's opened (issue #535).
            project.dirty = False
            return project

        example_projects.setdefault(section, {})[name] = wrapped
        return wrapped

    return decorator
