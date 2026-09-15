if not exist "%PREFIX%\Library\bin" mkdir "%PREFIX%\Library\bin" || exit /b 1
copy bin\check_cuda_arch.py "%PREFIX%\Library\bin\" || exit /b 1
copy bin\trim_cuda_archs.py "%PREFIX%\Library\bin\" || exit /b 1
