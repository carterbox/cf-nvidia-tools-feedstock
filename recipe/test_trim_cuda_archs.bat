setlocal

rem trim_cuda_archs.py bundles its own unit test suite; run it as an install-time check
python -m unittest cf_nvidia_tools.trim_cuda_archs -v
if errorlevel 1 exit /b 1

rem exercise the documented cmd.exe usage pattern end-to-end
set "CUDAARCHS=75-real;80-real;90a-real"
for /f "delims=" %%i in ('trim-cuda-archs CUDAARCHS 8.0') do set "CUDAARCHS=%%i"
if not "%CUDAARCHS%"=="80-real;90a-real" exit /b 1

rem bad arch exits 2
trim-cuda-archs CUDAARCHS not-an-arch >nul 2>&1
if not "%ERRORLEVEL%"=="2" exit /b 1

rem check_cuda_arch.py bundles its own unit test suite; run it as an install-time check
python -m unittest cf_nvidia_tools.check_cuda_arch -v
if errorlevel 1 exit /b 1

rem check_cuda_arch.py's own argument validation
check-cuda-arch >nul 2>&1
if not "%ERRORLEVEL%"=="2" exit /b 1

rem sanity check against a real redistributable library: libcublas (CUDA 13) still ships
rem SASS for sm_75, so the computed lowest common CUDA architecture must be 7.5
set "cuda_arch_version=7.5"
check-cuda-arch "%CONDA_PREFIX%\Library\bin\cublas64_*.dll"
if errorlevel 1 exit /b 1
