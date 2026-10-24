"""File-level smoke contract for the GitHub Pages hosted application."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _required_file(relative_path: str) -> Path:
    path = ROOT / relative_path
    assert path.is_file(), (
        f"GitHub Pages integration prerequisite is missing: {relative_path}. "
        "The app and deployment workflow have not been implemented yet."
    )
    return path


def test_vite_application_has_build_entry_and_pages_workflow():
    package = _required_file("package.json").read_text()
    entry = _required_file("index.html").read_text()
    vite_config = _required_file("vite.config.ts").read_text()
    workflow = _required_file(".github/workflows/deploy-pages.yml").read_text()

    assert '"build"' in package and "vite build" in package, (
        "package.json must expose the Vite production build used by Pages."
    )
    assert "<div" in entry and "<script" in entry, (
        "index.html must provide the application mount and module entry."
    )
    assert "base:" in vite_config, (
        "vite.config.ts must set a repository-subpath-safe Pages base URL."
    )
    assert "actions/deploy-pages" in workflow, (
        "The Pages workflow must publish the built static artifact."
    )
    assert "actions/upload-pages-artifact" in workflow, (
        "The Pages workflow must upload the production build artifact."
    )


def test_browser_entry_uses_subpath_safe_asset_references():
    entry = _required_file("index.html").read_text()
    # Vite rewrites relative module assets under its configured Pages base path.
    assert 'src="/src/' not in entry, (
        "Root-absolute /src assets break when GitHub Pages serves a project site."
    )
    assert 'href="/src/' not in entry, (
        "Root-absolute /src stylesheets break under a project-site subpath."
    )
