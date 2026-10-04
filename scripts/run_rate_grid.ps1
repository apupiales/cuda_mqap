# run_rate_grid.ps1
#
# Searches for the best configuration of population and greedy 2-opt rate for each instance.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_rate_grid.ps1
#
# How much of the population the local search improves is a compile-time constant, so the grid needs one
# binary per configuration: scripts\prepare_rates.py writes those trees and this script builds them. Then
# every instance runs at every population with every configuration, the same number of runs and the same
# seed, writing one result file per cell into -OutDir.
#
# The generation budget is fixed per family (KC10 70, KC20 and KC30 300), so the cells of one instance are
# comparable: what the grid answers is which configuration is best for a given budget, not how many
# generations an instance needs, which the convergence campaign measures.
#
# The run is resumable: a cell whose result file already exists is skipped.
#
# Copyright (C) 2019-2026 Andres Pupiales Arevalo <apupiales@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later

param(
    [string[]] $Instances      = @(),
    [string[]] $Populations    = @('1024', '4096', '16384', '65536'),
    [string[]] $Configurations = @('1.0', '0.5', '0.25', '0.1'),
    [int]      $Runs           = 10,
    [int]      $Seed           = 20260921,
    [string]   $OutDir         = 'results\grid',
    [string]   $BuildDir       = 'build\rates',
    [string]   $Arch           = 'sm_75',
    [string]   $VcVars         = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat',
    [switch]   $SkipBuild
)

$ErrorActionPreference = 'Continue'   # vcvars64.bat writes to stderr; the exit codes are checked instead
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# `powershell -File` hands an array over as one comma-joined string, so the lists are split again here.
function Split-List($values) { @($values | ForEach-Object { $_ -split ',' } | Where-Object { $_ }) }
$Instances      = Split-List $Instances
$Configurations = Split-List $Configurations
$Populations    = Split-List $Populations | ForEach-Object { [int]$_ }

# Every instance of mQAPData, unless the caller names some.
if (-not $Instances) {
    $Instances = Get-ChildItem (Join-Path $root 'mQAPData') -Filter '*.dat' |
        ForEach-Object { $_.BaseName } | Sort-Object
}

$generations = @{ 'KC10' = 70; 'KC20' = 300; 'KC30' = 300 }
$core = 'src\instance.cpp src\fitness.cu src\nsga2.cu src\nsga2_multiblock.cu src\operators.cu src\local_search.cu src\solver.cu'

New-Item -ItemType Directory -Force $OutDir | Out-Null

# One binary per configuration, from the current source of the branch.
$names = python scripts\prepare_rates.py $BuildDir @Configurations --list | ForEach-Object { ($_ -split '\s+')[0] }
if ($LASTEXITCODE -ne 0) { throw 'prepare_rates.py --list failed' }

if (-not $SkipBuild) {
    "### building $($names.Count) configurations"
    python scripts\prepare_rates.py $BuildDir @Configurations
    if ($LASTEXITCODE -ne 0) { throw 'prepare_rates.py failed' }
    foreach ($name in $names) {
        $tree = (Resolve-Path (Join-Path $BuildDir $name)).Path
        $build = "call ""$VcVars"" >nul 2>&1 && cd /d ""$tree"" && nvcc -O3 -arch=$Arch -std=c++17 -Xcompiler ""/Zc:preprocessor"" -Iinclude $core src\main.cpp -o ""$tree\cuda_mqap.exe"""
        & cmd /c $build 2>&1 | Select-String -Pattern 'error' | ForEach-Object { $_.Line }
        if (-not (Test-Path "$tree\cuda_mqap.exe")) { throw "$name : build failed" }
        "  $name ok"
    }
}

$total = $Instances.Count * $Populations.Count * $names.Count
$done = 0
$watch = [Diagnostics.Stopwatch]::StartNew()
"### $total cells: $($Instances.Count) instances x $($Populations.Count) populations x $($names.Count) configurations, $Runs runs each"

foreach ($instance in $Instances) {
    $family = $instance.Substring(0, 4)
    if (-not $generations.ContainsKey($family)) { throw "$instance : no generation budget for family $family" }
    $iterations = $generations[$family]
    $dir = Join-Path $OutDir $instance
    New-Item -ItemType Directory -Force $dir | Out-Null

    foreach ($population in $Populations) {
        foreach ($name in $names) {
            $done++
            $file = Join-Path $dir "${name}_P$population.txt"
            if (Test-Path $file) { continue }
            $exe = Join-Path (Join-Path $BuildDir $name) 'cuda_mqap.exe'
            $cell = [Diagnostics.Stopwatch]::StartNew()
            $text = & $exe "mQAPData\$instance.dat" --population $population --iterations $iterations `
                --runs $Runs --seed $Seed --verify --quiet --output $file 2>&1 | Out-String
            $cell.Stop()
            if ($text -notmatch 'Verification: OK') {
                "  !! $instance $name P=$population : verification did not report OK"
            }
            "[{0,4}/{1}] {2,-15} {3,-10} P={4,-6} {5,6:N1} s   (elapsed {6:N1} min)" -f `
                $done, $total, $instance, $name, $population, $cell.Elapsed.TotalSeconds, $watch.Elapsed.TotalMinutes
        }
    }
}
$watch.Stop()
"### done in $([math]::Round($watch.Elapsed.TotalHours, 2)) h"
