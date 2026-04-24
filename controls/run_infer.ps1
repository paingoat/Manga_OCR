param(
    [ValidateSet("crnn", "svtr")]
    [string]$Model = "crnn",
    [string]$Config = "",
    [string]$InputDir = "input/bubble",
    [string]$OutputRoot = "output",
    [string]$PaddleOCRDir = "PaddleOCR",
    [string]$CondaEnv = "pp_ocr_jap_infer",
    [switch]$UseGpu
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

if ([string]::IsNullOrWhiteSpace($Config)) {
    $Config = switch ($Model) {
        "crnn" { "configs/infer.crnn.yaml" }
        "svtr" { "configs/infer.svtr.yaml" }
    }
}

$ArgsList = @(
    "-m", "app.infer",
    "--config", $Config,
    "--input", $InputDir,
    "--output-root", $OutputRoot,
    "--paddleocr-dir", $PaddleOCRDir
)

if ($UseGpu) {
    $ArgsList += "--use-gpu"
}

conda run -n $CondaEnv python @ArgsList

