$ErrorActionPreference = "Stop"

Write-Host "=== Full TVM setup (Windows, source build) ==="

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$TVMRoot = Join-Path $ProjectRoot "apache-tvm-src"
$BuildDir = Join-Path $TVMRoot "build"
$VenvPath = Join-Path $ProjectRoot ".venv310"
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    throw "Python venv not found at $VenvPath . Please create .venv310 first."
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git is required."
}
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    throw "cmake is required."
}

if (-not (Test-Path $TVMRoot)) {
    Write-Host "Cloning apache/tvm..."
    git clone --recursive https://github.com/apache/tvm $TVMRoot
} else {
    Write-Host "TVM source already exists, updating submodules..."
    git -C $TVMRoot submodule update --init --recursive
}

if (-not (Test-Path $BuildDir)) {
    New-Item -Path $BuildDir -ItemType Directory | Out-Null
}

$ConfigTemplate = Join-Path $TVMRoot "cmake\config.cmake"
$ConfigFile = Join-Path $BuildDir "config.cmake"
Copy-Item $ConfigTemplate $ConfigFile -Force

Add-Content $ConfigFile "set(CMAKE_BUILD_TYPE RelWithDebInfo)"
Add-Content $ConfigFile "set(HIDE_PRIVATE_SYMBOLS ON)"
Add-Content $ConfigFile "set(USE_LLVM ON)"
Add-Content $ConfigFile "set(USE_METAL OFF)"
Add-Content $ConfigFile "set(USE_CUDA OFF)"
Add-Content $ConfigFile "set(USE_OPENCL OFF)"
Add-Content $ConfigFile "set(USE_VULKAN OFF)"

Write-Host "Building TVM C++ runtime/compiler..."
cmake -S $TVMRoot -B $BuildDir
cmake --build $BuildDir --config Release --parallel

Write-Host "Installing Python TVM package into .venv310..."
& $PythonExe -m pip install --upgrade pip setuptools wheel
& $PythonExe -m pip uninstall -y tlcpack-nightly apache-tvm-ffi mlc-ai-nightly-cpu
& $PythonExe -m pip install -e $TVMRoot

Write-Host "Installing notebook dependencies..."
& $PythonExe -m pip install numpy matplotlib torch torchvision Pygments

Write-Host ""
Write-Host "Done."
Write-Host "If import tvm fails, set TVM_LIBRARY_PATH to: $BuildDir"
