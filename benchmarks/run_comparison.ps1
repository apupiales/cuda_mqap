<#
    run_comparison.ps1

    Measures how much of the GPU each version of the program actually uses, with the same workload:
    the instance, the population and the number of generations compiled into kernel.cu of this branch.

    For every version it does two runs:
      1. a clean run, to take the wall time and the CPU time of the process,
      2. a run under Nsight Systems (nsys), to take the CUDA trace.
    analyze.py then turns the traces into the tables documented in README.md / LEEME.md.

    Usage (from a normal PowerShell; Visual Studio and the CUDA Toolkit must be installed):

        git worktree add ../cuda_mqap_opt        develop_with_claude_opus_5
        git worktree add ../cuda_mqap_p512       develop_p512_single_block
        git worktree add ../cuda_mqap_multiblock develop_large_population_multiblock
        .\benchmarks\run_comparison.ps1

    Copyright (C) 2019-2026 Andres Pupiales Arevalo <apupiales@gmail.com>
    SPDX-License-Identifier: GPL-3.0-or-later

    This program is free software: you can redistribute it and/or modify it under the terms of the
    GNU General Public License as published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version. This program is distributed in the hope that it
    will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should
    have received a copy of the GNU General Public License along with this program. If not, see
    <https://www.gnu.org/licenses/>.
#>
[CmdletBinding()]
param(
    # Source tree of each version. The original one is this branch.
    [string] $Original   = (Resolve-Path "$PSScriptRoot\.."),
    [string] $Optimized  = "$PSScriptRoot\..\..\cuda_mqap_opt",
    [string] $P512       = "$PSScriptRoot\..\..\cuda_mqap_p512",
    [string] $Multiblock = "$PSScriptRoot\..\..\cuda_mqap_multiblock",
    # Where the binaries, the traces and the CSV reports are written.
    [string] $Out        = "$PSScriptRoot\results",
    # Overrides, only needed if the tools are not where the script looks for them.
    [string] $Nsys       = '',
    [string] $VcVars     = '',
    # GPU architecture the four versions are compiled for (sm_75 = Turing, an RTX 2060).
    [string] $Arch       = 'sm_75',
    [switch] $SkipBuild
)

$ErrorActionPreference = 'Stop'

function Fail($message) { Write-Error $message; exit 1 }

# ---------------------------------------------------------------- tools

if (-not $VcVars) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $VcVars = & $vswhere -latest -products * -find VC\Auxiliary\Build\vcvars64.bat | Select-Object -First 1
    }
}
if (-not $VcVars -or -not (Test-Path $VcVars)) {
    Fail "vcvars64.bat not found. Pass it with -VcVars ""<path>\VC\Auxiliary\Build\vcvars64.bat""."
}

if (-not $Nsys) {
    $candidate = Get-Command nsys.exe -ErrorAction SilentlyContinue
    if ($candidate) {
        $Nsys = $candidate.Source
    } else {
        $Nsys = (Get-ChildItem 'C:\Program Files\NVIDIA Corporation\Nsight Systems*\target-windows-x64\nsys.exe' `
                 -ErrorAction SilentlyContinue | Sort-Object FullName | Select-Object -Last 1).FullName
    }
}
if (-not $Nsys -or -not (Test-Path $Nsys)) {
    Fail "nsys.exe not found (it ships with the CUDA Toolkit / Nsight Systems). Pass it with -Nsys ""<path>\nsys.exe""."
}

$python = (Get-Command python.exe -ErrorAction SilentlyContinue).Source
if (-not $python) { Fail 'python.exe not found in PATH (needed by analyze.py).' }

# ------------------------------------------------- workload of the comparison

# The original has its parameters compiled in: read them so that every version runs the same workload.
$kernel = Get-Content "$Original\kernel.cu" -Raw
$settingsName = ([regex]::Match($kernel, '(?m)^\s*#include\s+"(settings_[A-Za-z0-9_]+\.cu)"')).Groups[1].Value
if (-not $settingsName) { Fail "No active #include ""settings_*.cu"" found in $Original\kernel.cu." }
$settings = Get-Content "$Original\$settingsName" -Raw
function Get-Define($name) { [int]([regex]::Match($settings, "(?m)^\s*#define\s+$name\s+(\d+)")).Groups[1].Value }
$population = Get-Define 'POPULATION_SIZE'
$iterations = Get-Define 'ITERATIONS'
# settings_KC10_2fl_1rl.cu -> KC10-2fl-1rl
$instance = ($settingsName -replace '^settings_', '' -replace '\.cu$', '') -replace '_', '-'
$dat = "$Original\mQAPData\$instance.dat"
if (-not (Test-Path $dat)) { Fail "Instance file not found: $dat (derived from $settingsName)." }

Write-Host "Workload: $instance, population $population, $iterations generations, 1 run" -ForegroundColor Cyan

foreach ($pair in @(@('Optimized', $Optimized), @('P512', $P512), @('Multiblock', $Multiblock))) {
    if (-not (Test-Path "$($pair[1])\src\main.cpp")) {
        Fail ("$($pair[0]) source tree not found at $($pair[1]). Create it with, for example:`n" +
              "    git worktree add $($pair[1]) <branch>")
    }
}

# ---------------------------------------------------------------- build

$bin = "$Out\bin"
New-Item -ItemType Directory -Force -Path $bin | Out-Null

function Build($workdir, $sources, $exe, $extra) {
    # vcvars64.bat prints to stderr on some installations; only its output is silenced, not nvcc's.
    $command = "call ""$VcVars"" >nul 2>&1 && cd /d ""$workdir"" && nvcc -O3 -arch=$Arch -std=c++17 $extra $sources -o ""$exe"""
    # nvcc writes its warnings to stderr, which PowerShell would otherwise turn into a terminating error.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & cmd /c $command 2>&1 | Write-Host
    $code = $LASTEXITCODE
    $ErrorActionPreference = $previous
    if ($code -ne 0) { Fail "Build failed: $exe" }
}

$core = 'src\instance.cpp src\fitness.cu src\nsga2.cu src\operators.cu src\local_search.cu src\solver.cu src\main.cpp'
$coreMultiblock = $core -replace 'src\\nsga2\.cu', 'src\nsga2.cu src\nsga2_multiblock.cu'

if (-not $SkipBuild) {
    Write-Host '[1/4] original (this branch)'
    Build $Original 'kernel.cu' "$bin\orig.exe" ''
    Write-Host '[2/4] optimized'
    Build $Optimized $core "$bin\opt.exe" '-Iinclude'
    Write-Host '[3/4] p512'
    Build $P512 $core "$bin\p512.exe" '-Iinclude'
    Write-Host '[4/4] multiblock'
    # CUB, used by the multi-block survival, requires the conforming MSVC preprocessor.
    Build $Multiblock $coreMultiblock "$bin\mb.exe" '-Iinclude -Xcompiler "/Zc:preprocessor"'
}

# ---------------------------------------------------------------- cases

$common = @('--iterations', "$iterations", '--seed', '12345', '--quiet')
$cases = @(
    # Same workload for the four versions.
    @{ name = 'orig_P{0}'    -f $population; exe = "$bin\orig.exe"; args = @() },
    @{ name = 'opt_P{0}'     -f $population; exe = "$bin\opt.exe";  args = @($dat, '--population', "$population") + $common },
    @{ name = 'p512_P{0}'    -f $population; exe = "$bin\p512.exe"; args = @($dat, '--population', "$population") + $common },
    @{ name = 'mb_P{0}'      -f $population; exe = "$bin\mb.exe";   args = @($dat, '--population', "$population") + $common },
    # Largest population each branch supports, to show how the parallelism scales.
    @{ name = 'opt_P256';   exe = "$bin\opt.exe";  args = @($dat, '--population', '256')   + $common },
    @{ name = 'p512_P512';  exe = "$bin\p512.exe"; args = @($dat, '--population', '512')   + $common },
    @{ name = 'mb_P4096';   exe = "$bin\mb.exe";   args = @($dat, '--population', '4096')  + $common },
    @{ name = 'mb_P65536';  exe = "$bin\mb.exe";   args = @($dat, '--population', '65536') + $common }
)

$run = "$Out\run"
if (Test-Path $run) { Remove-Item -Recurse -Force $run }
New-Item -ItemType Directory -Force -Path "$run\rep" | Out-Null

function Run-Timed($exe, $arguments, $stdout) {
    $psi = New-Object Diagnostics.ProcessStartInfo
    $psi.FileName = $exe
    $psi.Arguments = ($arguments | ForEach-Object { if ($_ -match ' ') { '"' + $_ + '"' } else { $_ } }) -join ' '
    $psi.WorkingDirectory = Split-Path $stdout
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $p = [Diagnostics.Process]::Start($psi)
    $out = $p.StandardOutput.ReadToEndAsync()
    $err = $p.StandardError.ReadToEndAsync()
    $p.WaitForExit()
    Set-Content -Path $stdout -Value ($out.Result + $err.Result) -Encoding utf8
    [pscustomobject]@{
        wall = ($p.ExitTime - $p.StartTime).TotalSeconds
        cpu  = $p.TotalProcessorTime.TotalSeconds
        user = $p.UserProcessorTime.TotalSeconds
        sys  = $p.PrivilegedProcessorTime.TotalSeconds
        exit = $p.ExitCode
    }
}

# nsys reports its progress on stderr; do not let PowerShell turn that into a terminating error.
$ErrorActionPreference = 'Continue'

$rows = @()
foreach ($c in $cases) {
    $t = Run-Timed $c.exe $c.args "$run\$($c.name).out"
    if ($t.exit -ne 0) { Fail "$($c.name) exited with code $($t.exit); see $run\$($c.name).out" }
    $rep = "$run\rep\$($c.name)"
    $null = & $Nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true -o $rep $c.exe @($c.args) 2>&1
    $null = & $Nsys stats --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum --report cuda_api_sum `
                          --report cuda_gpu_trace --format csv --force-export=true --output $rep "$rep.nsys-rep" 2>&1
    $rows += [pscustomobject]@{
        name = $c.name
        wall = [math]::Round($t.wall, 3); cpu = [math]::Round($t.cpu, 3)
        user = [math]::Round($t.user, 3); sys = [math]::Round($t.sys, 3); exit = $t.exit
    }
    '{0,-12} wall={1,8:N3}s cpu={2,8:N3}s' -f $c.name, $t.wall, $t.cpu
}

# The CSV is read by analyze.py, which expects invariant (dot) decimal separators.
$rows | ForEach-Object {
    [pscustomobject]@{
        name = $_.name
        wall = $_.wall.ToString([Globalization.CultureInfo]::InvariantCulture)
        cpu  = $_.cpu.ToString([Globalization.CultureInfo]::InvariantCulture)
        user = $_.user.ToString([Globalization.CultureInfo]::InvariantCulture)
        sys  = $_.sys.ToString([Globalization.CultureInfo]::InvariantCulture)
        exit = $_.exit
    }
} | Export-Csv -Path "$run\times.csv" -NoTypeInformation -Encoding utf8

& $python "$PSScriptRoot\analyze.py" $run
if ($LASTEXITCODE -ne 0) { Fail 'analyze.py failed.' }
Write-Host "`nTraces and reports in $run" -ForegroundColor Cyan
