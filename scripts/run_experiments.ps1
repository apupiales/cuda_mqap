<#
.SYNOPSIS
    Runs the instances used in comparative_results_kcX_datasets.xlsx with the population and
    iterations of the former settings_*.cu files.

.EXAMPLE
    .\scripts\run_experiments.ps1 -Runs 30
    .\scripts\run_experiments.ps1 -Exe build\x64\Release\cuda_mqap.exe -Runs 10 -Seed 2026 -Instances KC10-2fl-1rl
#>
param(
    [string]$Exe = "build\x64\Release\cuda_mqap.exe",
    [int]$Runs = 10,
    [UInt64]$Seed = 0,
    [string]$OutDir = "results",
    [string[]]$Instances = @()
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

# Parameters of the former settings_<instance>.cu files. KC10-2fl-2uni used POPULATION_SIZE 4;
# the minimum population is now 16 (one warp per NSGA-II block needs 2P >= 32).
$experiments = [ordered]@{
    "KC10-2fl-1rl"  = @{ Population = 64;  Iterations = 70 }
    "KC10-2fl-1uni" = @{ Population = 16;  Iterations = 70 }
    "KC10-2fl-2rl"  = @{ Population = 16;  Iterations = 70 }
    "KC10-2fl-2uni" = @{ Population = 16;  Iterations = 70 }
    "KC10-2fl-3rl"  = @{ Population = 64;  Iterations = 70 }
    "KC10-2fl-3uni" = @{ Population = 128; Iterations = 25 }
    "KC10-2fl-4rl"  = @{ Population = 64;  Iterations = 70 }
    "KC10-2fl-5rl"  = @{ Population = 64;  Iterations = 70 }
    "KC20-2fl-1rl"  = @{ Population = 64;  Iterations = 300 }
    "KC20-2fl-1uni" = @{ Population = 64;  Iterations = 300 }
    "KC20-2fl-2uni" = @{ Population = 64;  Iterations = 300 }
    "KC20-2fl-3uni" = @{ Population = 64;  Iterations = 300 }
    "KC30-3fl-1rl"  = @{ Population = 32;  Iterations = 70 }
    "KC30-3fl-1uni" = @{ Population = 32;  Iterations = 70 }
    "KC30-3fl-2uni" = @{ Population = 32;  Iterations = 70 }
}

$exePath = if ([System.IO.Path]::IsPathRooted($Exe)) { $Exe } else { Join-Path $root $Exe }
if (-not (Test-Path $exePath)) {
    throw "Executable not found: $exePath (build the Release configuration first)"
}
$outPath = if ([System.IO.Path]::IsPathRooted($OutDir)) { $OutDir } else { Join-Path $root $OutDir }
New-Item -ItemType Directory -Force $outPath | Out-Null

$selected = if ($Instances.Count -gt 0) { $Instances } else { $experiments.Keys }
foreach ($name in $selected) {
    if (-not $experiments.Contains($name)) {
        throw "Unknown instance $name"
    }
    $p = $experiments[$name]
    $arguments = @(
        (Join-Path $root "mQAPData\$name.dat"),
        "--population", $p.Population,
        "--iterations", $p.Iterations,
        "--runs", $Runs,
        "--output", (Join-Path $outPath "result_${name}_nsga2_greedy_2opt.txt"),
        "--quiet"
    )
    if ($Seed -ne 0) {
        $arguments += @("--seed", $Seed)
    }
    & $exePath @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$name failed with exit code $LASTEXITCODE"
    }
}
