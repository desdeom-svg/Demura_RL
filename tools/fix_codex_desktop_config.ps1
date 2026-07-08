$ErrorActionPreference = "Stop"

$configPath = Join-Path $env:USERPROFILE ".codex\config.toml"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$backupPath = "$configPath.bak-$timestamp"

if (-not (Test-Path -LiteralPath $configPath)) {
    throw "Codex config not found: $configPath"
}

Copy-Item -LiteralPath $configPath -Destination $backupPath -Force

$content = Get-Content -LiteralPath $configPath -Raw
$content = $content -replace 'model_reasoning_effort\s*=\s*"high"', 'model_reasoning_effort = "medium"'
$content = $content -replace 'NODE_REPL_NATIVE_PIPE_CONNECT_TIMEOUT_MS\s*=\s*"1000"', 'NODE_REPL_NATIVE_PIPE_CONNECT_TIMEOUT_MS = "5000"'

Set-Content -LiteralPath $configPath -Value $content -Encoding UTF8

Write-Output "Updated: $configPath"
Write-Output "Backup:  $backupPath"
Get-Content -LiteralPath $configPath
