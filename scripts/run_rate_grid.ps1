# run_rate_grid.ps1
#
# Searches for the best configuration of population and greedy 2-opt rate for each instance.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_rate_grid.ps1
#
# Every instance runs at every population with every greedy configuration, the same number of runs and the
# same seed, writing one result file per cell into -OutDir. The configuration is set on the command line
# (--greedy-rate, --greedy-every), so one binary covers the whole grid, and -Untuned keeps the per-instance
# defaults of include/best_configuration.h out of the way: the grid is what measures them.
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
    [string]   $Exe            = 'build\x64\Release\cuda_mqap.exe',
    [string[]] $Instances      = @(),
    [string[]] $Populations    = @('1024', '4096', '16384', '65536'),
    [string[]] $Configurations = @('1.0', '0.5', '0.25', '0.1'),
    [int]      $Runs           = 10,
    [int]      $Seed           = 20260921,
    [string]   $OutDir         = 'results\grid'
)

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# `powershell -File` hands an array over as one comma-joined string, so the lists are split again here.
function Split-List($values) { @($values | ForEach-Object { $_ -split ',' } | Where-Object { $_ }) }
$Instances      = Split-List $Instances
$Configurations = Split-List $Configurations
$Populations    = Split-List $Populations | ForEach-Object { [int]$_ }

if (-not (Test-Path $Exe)) { throw "$Exe not found. Build this version first (see the README)." }

# Every instance of mQAPData, unless the caller names some.
if (-not $Instances) {
    $Instances = Get-ChildItem (Join-Path $root 'mQAPData') -Filter '*.dat' |
        ForEach-Object { $_.BaseName } | Sort-Object
}

$generations = @{ 'KC10' = 70; 'KC20' = 300; 'KC30' = 300 }
New-Item -ItemType Directory -Force $OutDir | Out-Null

# `rate:period` -> the name of the cell, which is also the label of the configuration in every report.
function Parse-Configuration($text) {
    $pieces = $text -split ':'
    $rate = [double]$pieces[0]
    $period = if ($pieces.Count -gt 1) { [int]$pieces[1] } else { 1 }
    if ($rate -lt 0 -or $rate -gt 1) { throw "the rate must be in [0, 1], got $($pieces[0])" }
    if ($period -lt 1) { throw "the period must be at least 1, got $period" }
    @{ rate = $rate; period = $period; name = "rate{0:d3}p{1}" -f [int][math]::Round($rate * 100), $period }
}

$cells = $Configurations | ForEach-Object { Parse-Configuration $_ }
$total = $Instances.Count * $Populations.Count * $cells.Count
$done = 0
$watch = [Diagnostics.Stopwatch]::StartNew()
"### $total cells: $($Instances.Count) instances x $($Populations.Count) populations x $($cells.Count) configurations, $Runs runs each"

foreach ($instance in $Instances) {
    $family = $instance.Substring(0, 4)
    if (-not $generations.ContainsKey($family)) { throw "$instance : no generation budget for family $family" }
    $iterations = $generations[$family]
    $dir = Join-Path $OutDir $instance
    New-Item -ItemType Directory -Force $dir | Out-Null

    foreach ($population in $Populations) {
        foreach ($cell in $cells) {
            $done++
            $file = Join-Path $dir "$($cell.name)_P$population.txt"
            if (Test-Path $file) { continue }
            $clock = [Diagnostics.Stopwatch]::StartNew()
            $text = & $Exe "mQAPData\$instance.dat" --population $population --iterations $iterations `
                --greedy-rate $cell.rate --greedy-every $cell.period --untuned `
                --runs $Runs --seed $Seed --verify --quiet --output $file 2>&1 | Out-String
            $clock.Stop()
            if ($text -notmatch 'Verification: OK') {
                "  !! $instance $($cell.name) P=$population : verification did not report OK"
            }
            "[{0,4}/{1}] {2,-15} {3,-10} P={4,-6} {5,6:N1} s   (elapsed {6:N1} min)" -f `
                $done, $total, $instance, $cell.name, $population, $clock.Elapsed.TotalSeconds, $watch.Elapsed.TotalMinutes
        }
    }
}
$watch.Stop()
"### done in $([math]::Round($watch.Elapsed.TotalHours, 2)) h"
