param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("on", "off")]
    [string]$Action,

    [string]$DemuraDir = "E:\WT\Projects\Demura AI\DemuraAIDemo\DemuraAIDemo\bin\Debug",
    [int]$Channel = 1,
    [int]$ReconnectWaitSeconds = 6,
    [switch]$NoRestart
)

$ErrorActionPreference = "Stop"

$exePath = Join-Path $DemuraDir "DemuraAIDemo.exe"
$managerDll = Join-Path $DemuraDir "DemuraPGManager.dll"
$configPath = Join-Path $DemuraDir "Config\Tester.xml"

if (-not (Test-Path -LiteralPath $exePath)) {
    throw "DemuraAIDemo.exe not found: $exePath"
}
if (-not (Test-Path -LiteralPath $managerDll)) {
    throw "DemuraPGManager.dll not found: $managerDll"
}
if (-not (Test-Path -LiteralPath $configPath)) {
    throw "Tester.xml not found: $configPath"
}

function Stop-DemuraAIDemo {
    $processes = Get-Process -Name DemuraAIDemo -ErrorAction SilentlyContinue
    if ($processes) {
        $processes | Stop-Process -Force
        Start-Sleep -Seconds 2
    }
}

function Start-DemuraAIDemo {
    Start-Process -FilePath $exePath -WorkingDirectory $DemuraDir -WindowStyle Hidden
}

Set-Location $DemuraDir

# The running DemuraAIDemo owns port 50001. Stop it briefly, create the same
# PGManager used by the UI button, execute the button action, then restore it.
Stop-DemuraAIDemo

[System.Reflection.Assembly]::LoadFrom($managerDll) | Out-Null
[System.Reflection.Assembly]::LoadFrom($exePath) | Out-Null

[DemuraAIDemo.PGManager]::m_ConfigFoler = $configPath
$pg = New-Object DemuraAIDemo.PGManager
$pg.Init([DemuraPGManager.PGType]::W6T)

Write-Output "PGManager initialized; waiting for WAgent reconnect..."
Start-Sleep -Seconds $ReconnectWaitSeconds

if ($Action -eq "off") {
    $result = $pg.PowerOffByChn($Channel)
    Write-Output "PowerOffByChn($Channel)=$result"
    if (-not $result) {
        throw "PowerOffByChn($Channel) returned false"
    }
}
else {
    $result = $pg.PowerOnByChn($Channel)
    Write-Output "PowerOnByChn($Channel)=$result"
    if ($result -ne 1) {
        throw "PowerOnByChn($Channel) returned $result"
    }
}

if (-not $NoRestart) {
    Start-DemuraAIDemo
    Start-Sleep -Seconds 4
}

