# run_original_comparison.ps1
#
# Runs the original version and this one on the same instances, many times each, and reports whether the
# difference between their fronts is statistically significant.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_original_comparison.ps1
#
# The original version has the instance and the number of repetitions compiled in, so it needs one build
# per instance: scripts\prepare_original.py writes that tree from the `master` branch, generating the
# settings file from the instance's own .dat and applying the memory fixes the code needs under CUDA 13.4.
# Both versions then run with the same population, generations and number of runs, and
# scripts\compare_versions.py measures both against the same reference front.
#
# The default configuration is the one the original version ships with, P = 64 and 300 generations, which
# is the only one where the two can be compared directly. -BigPopulation adds a third group with the
# population this version allows, at the same number of generations.
#
# Copyright (C) 2019-2026 Andres Pupiales Arevalo <apupiales@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later

param(
    [string]   $Exe         = 'build\x64\Release\cuda_mqap.exe',
    [string[]] $Instances   = @('KC20-2fl-1rl', 'KC20-2fl-1uni', 'KC20-2fl-2uni', 'KC20-2fl-3uni'),
    [int]      $Population  = 64,
    [int]      $Iterations  = 300,
    [int]      $Runs        = 30,
    [int]      $BigPopulation = 0,
    [int]      $Seed        = 20260922,
    [string]   $OutDir      = 'results\comparison',
    [string]   $OriginalRef = 'master',
    [string]   $Arch        = 'sm_75',
    [string]   $VcVars      = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat',
    [switch]   $SkipOriginal,
    [switch]   $SkipAnalysis,
    [switch]   $UpdateReference
)

$ErrorActionPreference = 'Continue'   # vcvars64.bat writes to stderr; the exit codes are checked instead
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path $Exe)) { throw "$Exe not found. Build this version first (see the README)." }
New-Item -ItemType Directory -Force $OutDir | Out-Null

foreach ($instance in $Instances) {
    $dir = Join-Path $OutDir $instance
    New-Item -ItemType Directory -Force $dir | Out-Null

    if (-not $SkipOriginal) {
        "### original version on $instance, $Runs runs of $Iterations generations with P = $Population"
        python scripts\prepare_original.py $instance $dir --ref $OriginalRef `
            --population $Population --iterations $Iterations --runs $Runs
        if ($LASTEXITCODE -ne 0) { throw "prepare_original.py failed on $instance" }

        $full = (Resolve-Path $dir).Path
        $build = "call ""$VcVars"" >nul 2>&1 && cd /d ""$full"" && nvcc -O3 -arch=$Arch -std=c++17 kernel.cu -o ""$full\original.exe"""
        & cmd /c $build 2>&1 | Select-String -Pattern 'error' | ForEach-Object { $_.Line }
        if (-not (Test-Path "$full\original.exe")) { throw "the build of the original version failed on $instance" }

        # The original version writes its result file into the working directory, with a fixed name.
        Get-ChildItem "$full\result_*.txt" -ErrorAction SilentlyContinue | Remove-Item
        $watch = [Diagnostics.Stopwatch]::StartNew()
        Push-Location $full
        & "$full\original.exe" *> "$full\original.log"
        Pop-Location
        $watch.Stop()
        $file = Get-ChildItem "$full\result_*.txt" -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $file) { throw "the original version wrote no result file on $instance (see original.log)" }
        $blocks = (Select-String -Path $file.FullName -Pattern '^\{' -AllMatches).Count
        "  $blocks runs in $([math]::Round($watch.Elapsed.TotalMinutes, 1)) min -> $($file.Name)"
    }

    "### this version on $instance, $Runs runs of $Iterations generations with P = $Population"
    $out = Join-Path $dir 'this_version.txt'
    if (Test-Path $out) { Remove-Item $out }
    & $Exe "mQAPData\$instance.dat" --population $Population --iterations $Iterations `
        --runs $Runs --seed $Seed --quiet --output $out 2>&1 |
        Select-String -Pattern 'Time Spent|Error' | ForEach-Object { "  $($_.Line)" }

    if ($BigPopulation -gt 0) {
        "### this version on $instance with P = $BigPopulation, same $Iterations generations"
        $big = Join-Path $dir 'this_version_big.txt'
        if (Test-Path $big) { Remove-Item $big }
        & $Exe "mQAPData\$instance.dat" --population $BigPopulation --iterations $Iterations `
            --runs $Runs --seed $Seed --quiet --output $big 2>&1 |
            Select-String -Pattern 'Time Spent|Error' | ForEach-Object { "  $($_.Line)" }
    }
}

if ($SkipAnalysis) { return }

foreach ($instance in $Instances) {
    $dir = Join-Path $OutDir $instance
    $groups = @()
    $original = Get-ChildItem "$dir\result_*.txt" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($original) { $groups += "original, P = $Population=$($original.FullName)" }
    if (Test-Path "$dir\this_version.txt") { $groups += "this version, P = $Population=$dir\this_version.txt" }
    if (Test-Path "$dir\this_version_big.txt") { $groups += "this version, P = $BigPopulation=$dir\this_version_big.txt" }
    if ($groups.Count -lt 2) { "  $instance : fewer than two result files, nothing to compare"; continue }

    $arguments = @('scripts\compare_versions.py', $instance) + $groups
    if ($UpdateReference) { $arguments += '--update-reference' }
    ''
    python @arguments
}
