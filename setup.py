from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class F2PYBuildExt(build_ext):
    def build_extension(self, ext: Extension) -> None:
        if ext.name != "dmrtml_for":
            super().build_extension(ext)
            return

        project_root = Path(__file__).resolve().parent
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        # NumPy f2py uses Meson on Python>=3.12 and stores configuration under
        # <build-dir>/bbdir. That configuration can hard-code absolute paths to
        # tools inside an isolated build environment. Wipe it to avoid reusing a
        # stale configuration when the build environment changes.
        bbdir = build_temp / "bbdir"
        if bbdir.exists():
            shutil.rmtree(bbdir, ignore_errors=True)

        # Where setuptools expects the built extension to land.
        ext_fullpath = Path(self.get_ext_fullpath(ext.name))
        ext_fullpath.parent.mkdir(parents=True, exist_ok=True)

        # Where editable installs expose sources from.
        # Setuptools editable typically adds the project "src" directory to
        # sys.path via a .pth file; placing the compiled extension there makes
        # `import walomis_for` work in editable mode.
        source_tree_ext_dir = project_root / "src"
        source_tree_ext_dir.mkdir(parents=True, exist_ok=True)

        # Build the f2py extension.
        # Mirrors the existing Makefile's `pydmrt` target.
        cmd = [
            sys.executable,
            "-m",
            "numpy.f2py",
            "-c",
            "--build-dir",
            str(build_temp),
            "-I.",
            "-llapack",
            "-lblas",
            "--f90flags=-O",
            "--f77flags=-O",
            "-m",
            "dmrtml_for",
            "src/dmrtml.pyf",
            "src/czergg.f",
            "src/dielectric_constant.f90",
            "src/dmrtparameters.F90",
            "src/fresnel.f90",
            "src/soil.f90",
            "src/disort.F90",
            "src/dmrtml.f90",
            "src/options.f90",
            "src/main.f90",

            # optimisation
            "--opt=-O3",
        ]

        subprocess.check_call(cmd, cwd=str(project_root))

        # f2py may write the extension into various locations depending on the
        # backend (distutils/meson), Python version, and whether the build is
        # editable. Be liberal in what we accept.
        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        candidates: list[str] = []
        candidates.extend(glob.glob(str(source_tree_ext_dir / f"dmrtml_for*{ext_suffix}")))
        candidates.extend(glob.glob(str(project_root / f"dmrtml_for*{ext_suffix}")))
        candidates.extend(glob.glob(str(build_temp / f"**/dmrtml_for*{ext_suffix}"), recursive=True))

        if not candidates:
            raise RuntimeError(
                "f2py build succeeded but no dmrtml_for shared library was found "
                f"in {build_temp} or {project_root}"
            )

        # Prefer artifacts in the build directory.
        built_path = Path(sorted(candidates, key=lambda p: (str(project_root) in p, p))[0])

        shutil.copy2(built_path, ext_fullpath)

        # Also copy into the source tree for editable installs.
        source_tree_ext_path = source_tree_ext_dir / ext_fullpath.name
        shutil.copy2(built_path, source_tree_ext_path)

        # Clean up an in-tree artifact if one was produced.
        try:
            if built_path.parent == project_root and ext_fullpath != built_path:
                built_path.unlink(missing_ok=True)
        except Exception:
            pass


setup(
    ext_modules=[Extension("dmrtml_for", sources=[])],
    cmdclass={"build_ext": F2PYBuildExt},
)
