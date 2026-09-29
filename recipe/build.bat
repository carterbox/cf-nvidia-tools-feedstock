rem The source tree has no .git (rattler-build copies a plain file tree), so setuptools_scm
rem needs the version handed to it explicitly rather than deriving it from `git describe`.
set "SETUPTOOLS_SCM_PRETEND_VERSION=%PKG_VERSION%"
%PYTHON% -m pip install --no-deps --no-build-isolation -vv .
if errorlevel 1 exit /b 1
