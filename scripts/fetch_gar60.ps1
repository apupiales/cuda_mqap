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
    Downloads the Gar60 mQAP instances (n = 60) into data\gar60, which git ignores.

.DESCRIPTION
    The 22 instances of Garrett and Dasgupta (2009), regenerated with the Knowles-Corne generator and
    published with PasMoQAP (Sanhueza et al., CEC 2017) at https://github.com/DataWaveAnalytics/pasmoqap.
    That repository declares no license, so the files are not redistributed here: this script fetches
    them. Only the 2- and 3-objective ones are downloaded by default, because the kernels support m <= 3
    and the 4-objective ones would not fit in 64 KB of shared memory at n = 60 anyway.

.EXAMPLE
    .\scripts\fetch_gar60.ps1
    .\scripts\fetch_gar60.ps1 -All
#>
param(
    [string]$OutDir = "data\gar60",
    [switch]$All
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$out = if ([System.IO.Path]::IsPathRooted($OutDir)) { $OutDir } else { Join-Path $root $OutDir }
New-Item -ItemType Directory -Force $out | Out-Null

$base = "https://raw.githubusercontent.com/DataWaveAnalytics/pasmoqap/master/data"
$names = @()
foreach ($i in 1..5) { $names += "Gar60-2fl-${i}uni"; $names += "Gar60-2fl-${i}rl" }
foreach ($i in 1..3) { $names += "Gar60-3fl-${i}uni"; $names += "Gar60-3fl-${i}rl" }
if ($All) { foreach ($i in 1..3) { $names += "Gar60-4fl-${i}uni"; $names += "Gar60-4fl-${i}rl" } }

foreach ($name in $names) {
    $target = Join-Path $out "$name.dat"
    if (Test-Path $target) { continue }
    Invoke-WebRequest -Uri "$base/$name.dat" -OutFile $target -UseBasicParsing
    Write-Host "downloaded $name.dat"
}
Write-Host "$($names.Count) instances in $out"
