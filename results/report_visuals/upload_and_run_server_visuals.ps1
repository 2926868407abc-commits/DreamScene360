$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Resolve-Path (Join-Path $scriptDir "..\..")

$ssh = "ssh"
$scp = "scp"
$remoteRoot = "/mnt/data/wangqq/DreamScene360"
$remoteVisualDir = "$remoteRoot/results/report_visuals"

Write-Host "[1/4] Create remote output directory"
& $ssh wangqq "mkdir -p $remoteVisualDir"

Write-Host "[2/4] Upload visualization scripts"
& $scp `
  (Join-Path $repo "scripts\visualize_report_depth_results.py") `
  "wangqq:$remoteRoot/scripts/"

& $scp `
  (Join-Path $repo "results\panorama_depth_three_dataset_summary.csv") `
  (Join-Path $scriptDir "run_server_depth_visuals.sh") `
  "wangqq:$remoteVisualDir/"

Write-Host "[3/4] Run server visualization generation"
& $ssh wangqq "cd $remoteRoot && bash results/report_visuals/run_server_depth_visuals.sh"

Write-Host "[4/4] Download generated figures"
& $scp "wangqq:$remoteVisualDir/*.png" "$scriptDir\"

Write-Host "[done] Figures are in $scriptDir"
