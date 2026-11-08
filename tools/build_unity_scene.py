#!/usr/bin/env python3
"""Build the Unity/Tuanjie hand scene from scratch, in one command.

Does everything the editor would otherwise require clicking through:

  1. finds the installed Tuanjie/Unity editor,
  2. copies the assets from this repo into the project,
  3. runs HandSceneBuilder.Build (model + 15 joints + rig + camera + light),
  4. renders a preview PNG so the result can be checked without opening the
     editor at all,
  5. reports what it did.

The editor must be CLOSED.  Unity takes an exclusive lock on a project, so a
second process cannot open the same one -- if it is open, this says so and
stops rather than failing with a stack trace.

Usage::

    python tools/build_unity_scene.py
    python tools/build_unity_scene.py --project D:\\hand_unity
    python tools/build_unity_scene.py --editor "C:/path/to/Tuanjie.exe"
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_PROJECT = Path("D:/hand_unity")

# Where Tuanjie Hub puts editors; the glob is ordered so the newest wins.
EDITOR_PATTERNS = [
    r"C:/Program Files/Tuanjie/Hub/Editor/*/Editor/Tuanjie.exe",
    r"C:/Program Files/Unity/Hub/Editor/*/Editor/Unity.exe",
    r"C:/Program Files/Unity*/Editor/Unity.exe",
]

BUILD_METHOD = "BionicHand.EditorTools.HandSceneBuilder.Build"
CAPTURE_METHOD = "BionicHand.EditorTools.HandSceneBuilder.CapturePreview"

# Folders copied from the repo into <project>/Assets.
ASSET_DIRS = ("Scripts", "Editor", "Models")


def find_editor(explicit: str = "") -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise SystemExit(f"[FAIL] editor not found: {path}")
        return path

    for pattern in EDITOR_PATTERNS:
        matches = sorted(glob.glob(pattern))
        if matches:
            return Path(matches[-1])

    raise SystemExit(
        "[FAIL] could not find a Tuanjie/Unity editor.\n"
        "       Pass --editor \"C:/path/to/Tuanjie.exe\"."
    )


def project_is_open(project: Path) -> bool:
    """Whether an editor currently holds this project.

    Checks the lock file first, then falls back to looking for an editor
    process launched with this -projectPath.  Tuanjie does not always leave
    Temp/UnityLockfile in place while a project is open, so the lock file alone
    reports "free" for a project that is very much in use -- and the build then
    dies with HandleProjectAlreadyOpenInAnotherInstance.
    """
    if (project / "Temp" / "UnityLockfile").exists():
        return True

    try:
        ps_command = (
            "Get-CimInstance Win32_Process -Filter \"Name='tuanjie.exe' or "
            "Name='Unity.exe'\" | Select-Object -ExpandProperty CommandLine"
        )
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_command],
            capture_output=True, text=True, timeout=30,
        )
        for line in completed.stdout.splitlines():
            if "-projectPath" in line and str(project) in line:
                return True
    except (OSError, subprocess.SubprocessError):
        pass
    return False


def copy_assets(project: Path) -> int:
    """Copy the repo's Unity assets into the project.  Returns files copied."""
    assets = project / "Assets"
    assets.mkdir(parents=True, exist_ok=True)

    copied = 0
    for name in ASSET_DIRS:
        source = REPO / "unity" / "Assets" / name
        if not source.exists():
            print(f"  [warn] {source} missing, skipped")
            continue
        target = assets / name
        # dirs_exist_ok keeps .meta files the project already generated, which
        # is what preserves script GUIDs across runs.
        shutil.copytree(source, target, dirs_exist_ok=True)
        copied += sum(1 for _ in target.rglob("*") if _.is_file())
        print(f"  copied {name}/")

    # Remove the single-file model, which Unity imports as one merged blob.
    merged = assets / "Models" / "hand.obj"
    if merged.exists():
        merged.unlink()
        meta = merged.with_suffix(".obj.meta")
        if meta.exists():
            meta.unlink()
        print("  removed hand.obj (per-part files are used instead)")
    return copied


def run_editor(editor: Path, project: Path, method: str, log: Path) -> int:
    command = [
        str(editor),
        "-batchmode",
        "-quit",
        "-projectPath", str(project),
        "-executeMethod", method,
        "-logFile", str(log),
    ]
    print(f"  running {method.split('.')[-1]} ...")
    completed = subprocess.run(command, capture_output=True, text=True)
    return completed.returncode


def tail_log(log: Path, needles, limit: int = 12):
    if not log.exists():
        return []
    hits = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if any(n in line for n in needles):
            hits.append(line.strip())
    return hits[-limit:]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the Unity hand scene in one command")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--editor", type=str, default="")
    parser.add_argument("--no-capture", action="store_true",
                        help="skip the preview render")
    parser.add_argument("--no-open", action="store_true",
                        help="do not launch the editor at the end")
    args = parser.parse_args()

    project = args.project
    if not project.exists():
        raise SystemExit(
            f"[FAIL] project not found: {project}\n"
            "       Create it once: Tuanjie Hub -> New Project -> 3D, "
            f"or pass --project."
        )

    if project_is_open(project):
        raise SystemExit(
            "[FAIL] the project is open in the editor.\n"
            "       Unity locks a project while it is open, so a second process "
            "cannot touch it.\n"
            "       Close the Tuanjie/Unity editor window (the Hub can stay "
            "open), then run this again."
        )

    editor = find_editor(args.editor)
    print(f"editor : {editor}")
    print(f"project: {project}")
    print()

    print("[1/4] copying assets")
    count = copy_assets(project)

    print("\n[2/4] building scene")
    build_log = Path("C:/TEMP/hand_build.log")
    code = run_editor(editor, project, BUILD_METHOD, build_log)
    for line in tail_log(build_log, ("[HandSceneBuilder]", "error CS", "Exception")):
        print(f"  {line[:150]}")

    scene = project / "Assets" / "Scenes" / "HandScene.unity"
    if not scene.exists():
        print(f"\n[FAIL] build did not produce {scene}")
        print(f"       full log: {build_log}")
        return 1

    print("\n[3/4] scene written")
    print(f"  {scene}  ({scene.stat().st_size / 1024:.0f} KB)")

    if not args.no_capture:
        print("\n[4/4] rendering preview")
        capture_log = Path("C:/TEMP/hand_capture.log")
        run_editor(editor, project, CAPTURE_METHOD, capture_log)
        for line in tail_log(capture_log, ("[Capture]", "error CS", "Exception")):
            print(f"  {line[:150]}")

        preview = project / "preview.png"
        if preview.exists():
            print(f"\n[OK] preview: {preview}  "
                  f"({preview.stat().st_size / 1024:.0f} KB)")
        else:
            print(f"\n[warn] no preview written; see {capture_log}")

    print("\nDone.")
    print("")
    print("  The scene is ready. To run the demo:")
    print("    1. the editor opens now -- wait for it to finish importing")
    print("    2. Project window -> Assets/Scenes -> double-click HandScene")
    print("    3. press Ctrl+P to play")
    print("    4. in another terminal:")
    print("         python tools/unity_pose_test.py send --rate 60 --hold 5 --ramp 2")

    if not args.no_open:
        print("\n[opening the editor]")
        # GUI launch, detached: this script must not wait for the editor, which
        # stays open until the user closes it.
        subprocess.Popen(
            [str(editor), "-projectPath", str(project)],
            close_fds=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
