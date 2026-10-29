from pathlib import Path


def test_executable_code_has_no_legacy_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    forbidden = ("artifacts/debug", "artifacts/exports", "/home/agent/", "/root/wja/")
    files = []
    for directory in ("src", "workflows", "extras"):
        files.extend((root / directory).rglob("*.py"))
        files.extend((root / directory).rglob("*.sh"))
    # This migration utility must recognize legacy prefixes in order to remove
    # them; it is not a runtime dependency or an embedded default path.
    files = [path for path in files if path.name != "sanitize_result_paths.py"]
    hits = [(path, token) for path in files for token in forbidden if token in path.read_text()]
    assert not hits
