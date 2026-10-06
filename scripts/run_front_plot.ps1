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
    Runs the default call of each instance, keeping its initial population, and plots the front it ends
    with against the best known front and that initial population (scripts/plot_fronts.py).

.DESCRIPTION
    The default call is the configuration measured best for the instance (include/best_configuration.h):
    on the KC30 instances, P = 65536 and 300 generations, one to two minutes of GPU per instance on an
    RTX 2060. Only one run per instance, because the figure shows one run. The run is verified on the CPU,
    the initial population included.

.EXAMPLE
    .\scripts\run_front_plot.ps1
    .\scripts\run_front_plot.ps1 -Instances all -Png
    .\scripts\run_front_plot.ps1 -Instances KC30-3fl-1rl -Seed 2026 -Png -SelfContained
    .\scripts\run_front_plot.ps1 -Instances KC20-2fl-1rl,KC10-2fl-1rl -OutDir results\other
#>
param(
    [string]$Exe = "build\x64\Release\cuda_mqap.exe",
    [UInt64]$Seed = 20261005,
    [string]$OutDir = "results\fronts",
    [string[]]$Instances = @(),
    [string]$Reference = "",
    [switch]$Png,
    [switch]$SelfContained
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

if ($Instances.Count -eq 0 -or $Instances -contains "all") {
    # By default the KC30 instances; "all" takes every .dat of mQAPData.
    $filter = if ($Instances -contains "all") { "*.dat" } else { "KC30-*.dat" }
    $Instances = Get-ChildItem (Join-Path $root "mQAPData") -Filter $filter | Sort-Object Name |
        ForEach-Object { $_.BaseName }
}
$exePath = if ([System.IO.Path]::IsPathRooted($Exe)) { $Exe } else { Join-Path $root $Exe }
if (-not (Test-Path $exePath)) {
    throw "Program not found: $exePath (build the Release configuration first)"
}
$out = if ([System.IO.Path]::IsPathRooted($OutDir)) { $OutDir } else { Join-Path $root $OutDir }
New-Item -ItemType Directory -Force $out | Out-Null

foreach ($instance in $Instances) {
    $data = Join-Path $root "mQAPData\$instance.dat"
    $result = Join-Path $out "${instance}_result.txt"
    $initial = Join-Path $out "${instance}_initial.csv"
    # The result file is appended to, and the figure plots its last block: start from an empty one.
    if (Test-Path $result) { Remove-Item $result }

    Write-Host "== $instance"
    # No --population, --iterations or --greedy-*: the default call of the instance, which the first
    # lines of the output describe and the title of the figure repeats.
    $output = & $exePath $data --seed $Seed --quiet --verify --output $result --initial $initial 2>&1
    $output | ForEach-Object { Write-Host "   $_" }
    if ($LASTEXITCODE -ne 0) {
        throw "$instance failed (exit code $LASTEXITCODE)"
    }
    $header = ($output | Select-String -Pattern '^Instance ' | Select-Object -First 1).Line
    $greedy = ($output | Select-String -Pattern '^Greedy 2-opt' | Select-Object -First 1).Line
    $config = if ($header -match 'population = (\d+), iterations = (\d+)') { "P = $($Matches[1]), $($Matches[2]) generations" } else { "" }
    if ($greedy -match '^Greedy 2-opt on ([^|]+?)\s*\|') { $config += ", greedy 2-opt on $($Matches[1])" }
    $label = "Default call: cuda_mqap $instance.dat --seed $Seed ($config)"

    $plot = @((Join-Path $PSScriptRoot "plot_fronts.py"), $instance, "--result", $result, "--initial", $initial,
              "--out", $out, "--label", $label)
    if ($Reference) { $plot += @("--reference", $Reference) }
    if ($Png) { $plot += "--png" }
    if ($SelfContained) { $plot += "--self-contained" }
    & python @plot
    if ($LASTEXITCODE -ne 0) {
        throw "plot_fronts.py failed on $instance"
    }
}
