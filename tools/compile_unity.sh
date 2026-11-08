#!/usr/bin/env bash
# Compile the Unity C# outside the editor.
#
# The editor is the authority, but it needs a licence and refuses to run while
# it already has the project open.  Its bundled Roslyn can type-check the same
# sources in a couple of seconds, which is enough to catch the ordinary
# mistakes before waiting on a full domain reload.
#
# Usage: bash tools/compile_unity.sh
set -u

EDITOR="${TUANJIE_EDITOR:-/c/Program Files/Tuanjie/Hub/Editor/2022.3.62t16/Editor}"
CSC="$EDITOR/Data/DotNetSdkRoslyn/csc.dll"
OUT="/c/TEMP/unitycheck"
mkdir -p "$OUT"

if [ ! -f "$CSC" ]; then
  echo "[FAIL] Roslyn not found at $CSC"
  echo "       set TUANJIE_EDITOR to the editor's Editor/ folder"
  exit 1
fi

W() { cygpath -w "$1"; }

REFS=()
add() { [ -f "$1" ] && REFS+=("-r:$(W "$1")"); }

add "$EDITOR/Data/NetStandard/ref/2.1.0/netstandard.dll"
for f in "$EDITOR"/Data/NetStandard/compat/2.1.0/shims/netfx/*.dll; do add "$f"; done

# Reference the monolithic UnityEngine.dll and NOT the per-module assemblies
# under Data/Managed/UnityEngine/.  The two overlap: pulling in both makes the
# compiler report every attribute as CS0433 ("exists in two assemblies"), and
# pulling in only the modules leaves Scene/SceneManager unresolved.
add "$EDITOR/Data/Managed/UnityEngine.dll"
add "$EDITOR/Data/Managed/UnityEditor.dll"

SOURCES=()
while IFS= read -r -d '' f; do SOURCES+=("$(W "$f")"); done \
  < <(find unity/Assets/Scripts unity/Assets/Editor -name '*.cs' -print0)

dotnet "$(W "$CSC")" -nologo -nostdlib -target:library \
  -out:"$(W "$OUT")/check.dll" "${REFS[@]}" "${SOURCES[@]}" 2>&1
