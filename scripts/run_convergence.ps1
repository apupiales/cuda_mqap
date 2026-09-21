# Copyright (C) 2019-2026 Andres Pupiales Arevalo <apupiales@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

<#
.SYNOPSIS
    Records with --trace the front of every generation of several runs per instance, and analyses the
    traces with scripts/analyze_convergence.py to find how many generations each instance needs.

.DESCRIPTION
    The cap of -Iterations has to be clearly above the generation where the search stagnates, or the
    measurement reports the cap instead of the stagnation. The analysis says so: when t_stall or
    t_final land on the last generation, raise -Iterations and run it again.

    Tracing copies the survivors to the host once per generation, so the times printed by these runs
    are not comparable with a normal run.

.EXAMPLE
    .\scripts\run_convergence.ps1
    .\scripts\run_convergence.ps1 -Population 4096 -Iterations 300 -Runs 30 -Instances KC10-2fl-1rl
#>
param(
    [string]$Exe = "build\x64\Release\cuda_mqap.exe",
    [int]$Population = 1024,
    [int]$Iterations = 200,
    [int]$Runs = 30,
    [int]$TraceMax = 4096,
    [int]$TraceEvery = 1,
    [UInt64]$Seed = 20260921,
    [string]$OutDir = "results\convergence",
    [string[]]$Instances = @(),
    [switch]$SkipAnalysis
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

# KC10 instances have a published .PO front, so for them the analysis also reports the generation at
# which the optimal front is reached; for the rest only the stagnation is measurable.
$default = @(
    "KC10-2fl-1rl", "KC10-2fl-1uni", "KC10-2fl-2rl", "KC10-2fl-2uni",
    "KC10-2fl-3rl", "KC10-2fl-3uni", "KC10-2fl-4rl", "KC10-2fl-5rl",
    "KC20-2fl-1rl", "KC20-2fl-1uni", "KC20-2fl-2uni", "KC20-2fl-3uni",
    "KC30-3fl-1rl", "KC30-3fl-1uni", "KC30-3fl-2uni"
)
$selected = if ($Instances.Count -gt 0) { $Instances } else { $default }

$exePath = if ([System.IO.Path]::IsPathRooted($Exe)) { $Exe } else { Join-Path $root $Exe }
if (-not (Test-Path $exePath)) {
    throw "Executable not found: $exePath. Build it first, or pass -Exe."
}
$outPath = if ([System.IO.Path]::IsPathRooted($OutDir)) { $OutDir } else { Join-Path $root $OutDir }
New-Item -ItemType Directory -Force -Path $outPath | Out-Null

$traces = @()
foreach ($name in $selected) {
    $dat = Join-Path $root "mQAPData\$name.dat"
    if (-not (Test-Path $dat)) {
        Write-Warning "Instance not found: $dat"
        continue
    }
    $trace = Join-Path $outPath "$name`_P$Population.csv"
    $result = Join-Path $outPath "$name`_P$Population`_result.txt"
    if (Test-Path $result) { Remove-Item $result }
    $watch = [Diagnostics.Stopwatch]::StartNew()
    # The program warns on stderr when a front did not fit in --trace-max; that must not stop the
    # campaign, so the stream is merged and only the exit code decides.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $exePath $dat --population $Population --iterations $Iterations --runs $Runs --seed $Seed `
        --quiet --trace $trace --trace-max $TraceMax --trace-every $TraceEvery --output $result 2>&1 |
        Where-Object { $_ -match 'Warning' } | ForEach-Object { Write-Warning "$name`: $_" }
    $code = $LASTEXITCODE
    $ErrorActionPreference = $previous
    $watch.Stop()
    if ($code -ne 0) { throw "$name failed with exit code $code" }
    $traces += $trace
    "{0,-16} P={1,-6} gens={2,-4} runs={3,-4} {4,7:N1}s  {5:N1} MB" -f `
        $name, $Population, $Iterations, $Runs, $watch.Elapsed.TotalSeconds, ((Get-Item $trace).Length / 1MB)
}

if (-not $SkipAnalysis -and $traces.Count -gt 0) {
    $python = (Get-Command python.exe -ErrorAction SilentlyContinue).Source
    if (-not $python) {
        Write-Warning "python.exe not found: run scripts/analyze_convergence.py yourself over $outPath"
    } else {
        ""
        & $python (Join-Path $PSScriptRoot "analyze_convergence.py") @traces `
            --po-dir (Join-Path $root "mQAPData") --out-dir (Join-Path $outPath "curves")
    }
}
