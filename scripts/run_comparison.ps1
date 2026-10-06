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
    Comparison of cuda_mqap with its CPU version and with pymoo's MOEAs, in four blocks.

.DESCRIPTION
    speedup  The same algorithm on the GPU and on the CPU (--cpu, OpenMP): time per generation at
             P = 64 ... 16384 on KC10-2fl-1rl, KC20-2fl-1rl and KC30-3fl-1rl, 20 generations, 3 runs each.
    budget   Same work: P = 64 and the generations of the original version (70 on KC10, 300 otherwise),
             30 runs, cuda_mqap --untuned against the six pymoo algorithms, on the 23 instances.
    gar60    The Gar60 instances (n = 60): the same-work comparison with 100 generations, plus cuda_mqap
             at P = 4096 with 100 generations against the pymoo algorithms given the same wall time.
    time     Same wall time: the default call of cuda_mqap on each of the 23 instances against the pymoo
             algorithms given, per run, the wall time of one run of that call.

    Every output goes to results\comparison\<block>\<instance>\<algorithm>.txt (the result file format,
    which scripts\compare_versions.py reads) with a .json or .log next to it. A finished output is not
    recomputed, so an interrupted campaign is resumed by running the same command again.

.EXAMPLE
    .\scripts\run_comparison.ps1 -Block speedup
    .\scripts\run_comparison.ps1 -Block budget,gar60,time
#>
param(
    [ValidateSet('speedup', 'budget', 'gar60', 'time')]
    [string[]]$Block = @('speedup', 'budget', 'gar60', 'time'),
    [string]$Exe = "build\x64\Release\cuda_mqap.exe",
    [int]$Runs = 30,
    [UInt64]$Seed = 20261006,
    [string]$OutDir = "results\comparison",
    [int]$Jobs = 6        # concurrent pymoo runs: one per physical core, so a time budget means the same
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$pymoo = Join-Path $PSScriptRoot "baselines\pymoo_mqap.py"
$algorithms = @('nsga2', 'nsga2-ls', 'nsga3', 'nsga3-ls', 'moead', 'moead-ls')
$kc = Get-ChildItem mQAPData -Filter "*.dat" | Sort-Object Name | ForEach-Object { $_.BaseName }
$gar = Get-ChildItem data\gar60 -Filter "*.dat" -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object { $_.BaseName }

function Invoke-Cuda([string]$data, [string]$output, [string[]]$options) {
    if (Test-Path $output) { return }
    $log = [System.IO.Path]::ChangeExtension($output, ".log")
    $partial = "$output.partial"
    if (Test-Path $partial) { Remove-Item $partial }
    $text = & $Exe $data --quiet --verify --output $partial @options 2>&1
    $text | Out-File $log -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw "cuda_mqap failed on $data ($options)" }
    Move-Item $partial $output
}

function Invoke-Pymoo([string]$data, [string]$output, [string[]]$options) {
    if (Test-Path $output) { return }
    $partial = "$output.partial.txt"
    & python $pymoo $data --output $partial --runs $Runs --seed $Seed --jobs $Jobs @options | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "pymoo_mqap.py failed on $data ($options)" }
    Move-Item $partial $output
    Move-Item ([System.IO.Path]::ChangeExtension($partial, ".json")) ([System.IO.Path]::ChangeExtension($output, ".json")) -Force
}

function Get-WallSeconds([string]$data, [string[]]$options) {
    # One run alone, as the time budget of one pymoo run.
    $text = & $Exe $data --quiet --runs 1 --seed $Seed --output "$env:TEMP\cuda_mqap_timing.txt" @options 2>&1
    Remove-Item "$env:TEMP\cuda_mqap_timing.txt" -ErrorAction SilentlyContinue
    return [double](($text | Select-String 'Time Spent: ([\d.]+) s').Matches[0].Groups[1].Value)
}

$generationsOf = { param($name) if ($name -like 'KC10-*') { 70 } elseif ($name -like 'Gar60-*') { 100 } else { 300 } }

foreach ($b in $Block) {
    $dir = Join-Path $OutDir $b
    New-Item -ItemType Directory -Force $dir | Out-Null
    Write-Host "==== block $b ($(Get-Date -Format s))"

    if ($b -eq 'speedup') {
        $csv = Join-Path $dir "speedup.csv"
        if (-not (Test-Path $csv)) { "instance,population,device,seed,generations,milliseconds" | Out-File $csv -Encoding utf8 }
        $done = Get-Content $csv | Select-Object -Skip 1
        foreach ($name in 'KC10-2fl-1rl', 'KC20-2fl-1rl', 'KC30-3fl-1rl') {
            foreach ($p in 64, 256, 1024, 4096, 16384) {
                foreach ($device in 'gpu', 'cpu') {
                    foreach ($r in 0..2) {
                        $s = $Seed + $r
                        if ($done -match "^$name,$p,$device,$s,") { continue }
                        # Built this way on purpose: "if (...) { @('--cpu') }" unrolls to a string, which
                        # splatting would not pass as one argument.
                        $extra = @()
                        if ($device -eq 'cpu') { $extra += '--cpu' }
                        $text = & $Exe "mQAPData\$name.dat" --untuned --population $p --iterations 20 --seed $s --quiet --output "$env:TEMP\cuda_mqap_speed.txt" @extra 2>&1
                        $ms = ($text | Select-String '(GPU|CPU) ([\d.]+) ms').Matches[0].Groups[2].Value
                        "$name,$p,$device,$s,20,$ms" | Out-File $csv -Append -Encoding utf8
                        Write-Host "  $name P=$p $device seed $s : $ms ms"
                    }
                }
            }
        }
    }

    if ($b -eq 'budget' -or $b -eq 'gar60') {
        $names = if ($b -eq 'budget') { $kc } else { $gar }
        $folder = if ($b -eq 'budget') { 'mQAPData' } else { 'data\gar60' }
        foreach ($name in $names) {
            $g = & $generationsOf $name
            $d = Join-Path $dir $name
            New-Item -ItemType Directory -Force $d | Out-Null
            Write-Host "-- $name, same work: P = 64, $g generations"
            Invoke-Cuda "$folder\$name.dat" (Join-Path $d "cuda_mqap_p64.txt") @('--untuned', '--population', '64', '--iterations', "$g", '--runs', "$Runs", '--seed', "$Seed")
            foreach ($a in $algorithms) {
                Invoke-Pymoo "$folder\$name.dat" (Join-Path $d "$a`_p64.txt") @('--algorithm', $a, '--pop', '64', '--gen', "$g")
            }
            if ($b -eq 'gar60') {
                # Same wall time: cuda_mqap at P = 4096, the pymoo algorithms with the time of one such run.
                $options = @('--untuned', '--population', '4096', '--iterations', "$g")
                Invoke-Cuda "$folder\$name.dat" (Join-Path $d "cuda_mqap_p4096.txt") ($options + @('--runs', "$Runs", '--seed', "$Seed"))
                $seconds = Get-WallSeconds "$folder\$name.dat" $options
                Write-Host "-- $name, same wall time: $seconds s per run"
                foreach ($a in $algorithms) {
                    Invoke-Pymoo "$folder\$name.dat" (Join-Path $d "$a`_time.txt") @('--algorithm', $a, '--pop', '100', '--seconds', "$seconds")
                }
            }
        }
    }

    if ($b -eq 'time') {
        foreach ($name in $kc) {
            $d = Join-Path $dir $name
            New-Item -ItemType Directory -Force $d | Out-Null
            # The default call: the configuration of include\best_configuration.h.
            Invoke-Cuda "mQAPData\$name.dat" (Join-Path $d "cuda_mqap_default.txt") @('--runs', "$Runs", '--seed', "$Seed")
            $seconds = Get-WallSeconds "mQAPData\$name.dat" @()
            Write-Host "-- $name, same wall time: $seconds s per run"
            foreach ($a in $algorithms) {
                Invoke-Pymoo "mQAPData\$name.dat" (Join-Path $d "$a`_time.txt") @('--algorithm', $a, '--pop', '100', '--seconds', "$seconds")
            }
        }
    }
}
Write-Host "==== done ($(Get-Date -Format s))"
