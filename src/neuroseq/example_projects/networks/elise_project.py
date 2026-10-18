from neuroseq.example_projects._registry import register_example_project
from neuroseq.project import Project


@register_example_project("ELISE Sequence Learning Network", section="Network")
def elise_project() -> Project:
    """Build the ELISE example using its backend's project factory."""
    from neuroseq.example_models.elise import create_project

    return create_project()
