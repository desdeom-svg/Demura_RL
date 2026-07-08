param(
    [string]$PythonExe = "E:\softWare\Anaconda\envs\Pytorch\python.exe",
    [int]$Gray = 16,
    [int]$Episodes = 400,
    [int]$Steps = 4,
    [int]$BatchSize = 2,
    [int]$BufferCapacity = 512,
    [int]$LearnCropSize = 200,
    [int]$PatchesPerTransition = 8,
    [int]$SliceGridRows = 10,
    [int]$SliceGridCols = 5,
    [string]$BootstrapPriorGains = "0.5,1.0",
    [int]$BootstrapMinPositive = 2,
    [int]$BootstrapMinNegative = 4,
    [double]$BootstrapPositiveRewardThreshold = 0.0005,
    [double]$BootstrapNegativeRewardThreshold = -0.005,
    [int]$CriticWarmupUpdates = 120,
    [int]$ActorWarmupEpisodes = 20,
    [int]$ActorUpdateEvery = 4,
    [double]$GainLimitInit = 0.05,
    [double]$GainLimitFinal = 0.30,
    [int]$GainLimitRampEpisodes = 100,
    [int]$GainLimitRampDelay = 20,
    [double]$GainAbsWeight = 0.005,
    [double]$GainTvWeight = 0.02,
    [double]$GainLaplacianWeight = 0.03,
    [double]$StepGainDecay = 0.25,
    [double]$LargeCropUpdateRatio = 0.0,
    [double]$ResidualClip = 0.05,
    [double]$ResidualNoiseInit = 0.008,
    [double]$ResidualNoiseMin = 0.005,
    [double]$QualityMeanWeight = 1.2,
    [double]$QualityP95Weight = 0.4,
    [double]$QualityTailWeight = 1.0,
    [double]$QualityProfileWeight = 0.8,
    [string]$ReferenceModelPath = "",
    [double]$ReferenceMetricGate = 1.0,
    [double]$MinReplayEffectiveActionAbsMean = 0.000001,
    [double]$MinReplayQuantizedStepChangedRatio = 0.000001,
    [int]$Patience = 250,
    [int]$BestPatience = 80,
    [int]$ConnectTimeoutSeconds = 30,
    [switch]$DryRun,
    [ValidateSet("Hidden", "Minimized", "Normal")]
    [string]$WindowMode = "Hidden",
    [switch]$ForceCpu
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$realWorldDir = Join-Path $repoRoot "RealWorld_Train"
$bootstrapLog = Join-Path $realWorldDir "live_runner_bootstrap.log"
$pidFile = Join-Path $realWorldDir "live_runner.pid"
$stdoutLog = Join-Path $realWorldDir "live_runner_stdout.log"
$stderrLog = Join-Path $realWorldDir "live_runner_stderr.log"
$bootstrapPy = Join-Path $PSScriptRoot "run_iterative_training_bootstrap.py"
$trainArgsFile = Join-Path $realWorldDir "live_runner_train_args.txt"

New-Item -ItemType Directory -Force -Path $realWorldDir | Out-Null
Set-Content -Path $stdoutLog -Value "" -Encoding utf8
Set-Content -Path $stderrLog -Value "" -Encoding utf8

$args = @(
    "train_real.py",
    "--gray", $Gray,
    "--episodes", $Episodes,
    "--steps", $Steps,
    "--batch-size", $BatchSize,
    "--buffer-capacity", $BufferCapacity,
    "--learn-crop-size", $LearnCropSize,
    "--patches-per-transition", $PatchesPerTransition,
    "--slice-grid-rows", $SliceGridRows,
    "--slice-grid-cols", $SliceGridCols,
    "--bootstrap-prior-gains", $BootstrapPriorGains,
    "--bootstrap-min-positive", $BootstrapMinPositive,
    "--bootstrap-min-negative", $BootstrapMinNegative,
    "--bootstrap-positive-reward-threshold", $BootstrapPositiveRewardThreshold,
    "--bootstrap-negative-reward-threshold", $BootstrapNegativeRewardThreshold,
    "--critic-warmup-updates", $CriticWarmupUpdates,
    "--actor-warmup-episodes", $ActorWarmupEpisodes,
    "--actor-update-every", $ActorUpdateEvery,
    "--gain-limit-init", $GainLimitInit,
    "--gain-limit-final", $GainLimitFinal,
    "--gain-limit-ramp-episodes", $GainLimitRampEpisodes,
    "--gain-limit-ramp-delay", $GainLimitRampDelay,
    "--gain-abs-weight", $GainAbsWeight,
    "--gain-tv-weight", $GainTvWeight,
    "--gain-laplacian-weight", $GainLaplacianWeight,
    "--step-gain-decay", $StepGainDecay,
    "--large-crop-update-ratio", $LargeCropUpdateRatio,
    "--residual-clip", $ResidualClip,
    "--residual-noise-init", $ResidualNoiseInit,
    "--residual-noise-min", $ResidualNoiseMin,
    "--quality-mean-weight", $QualityMeanWeight,
    "--quality-p95-weight", $QualityP95Weight,
    "--quality-tail-weight", $QualityTailWeight,
    "--quality-profile-weight", $QualityProfileWeight,
    "--reference-metric-gate", $ReferenceMetricGate,
    "--min-replay-effective-action-abs-mean", $MinReplayEffectiveActionAbsMean,
    "--min-replay-quantized-step-changed-ratio", $MinReplayQuantizedStepChangedRatio,
    "--patience", $Patience,
    "--best-patience", $BestPatience
)
if ($ReferenceModelPath -ne "") {
    $args += @("--reference-model-path", $ReferenceModelPath)
}
if ($DryRun) {
    $args += "--dry-run"
}

$quotedArgs = $args | ForEach-Object {
    $value = [string]$_
    '"' + ($value -replace '"', '\"') + '"'
}
$pythonCommand = '"' + $PythonExe + '" ' + ([string]::Join(' ', $quotedArgs))
[System.IO.File]::AppendAllText(
    $bootstrapLog,
    ("[{0}] Launching: {1}{2}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $pythonCommand, [Environment]::NewLine),
    [System.Text.Encoding]::UTF8
)
Set-Content -Path $trainArgsFile -Value $args -Encoding utf8
$env:DEMURA_SOCKET_TIMEOUT = [string]$ConnectTimeoutSeconds
if ($ForceCpu) {
    $env:CUDA_VISIBLE_DEVICES = "-1"
}
else {
    Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
}
$launcherArgs = @(
    $bootstrapPy,
    $bootstrapLog
) + $args

$process = Start-Process -FilePath $PythonExe -ArgumentList $launcherArgs -WorkingDirectory $repoRoot -WindowStyle $WindowMode -PassThru

Set-Content -Path $pidFile -Value $process.Id -Encoding ascii
[System.IO.File]::AppendAllText(
    $bootstrapLog,
    ("[{0}] Started python-bootstrap PID {1}{2}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $process.Id, [Environment]::NewLine),
    [System.Text.Encoding]::UTF8
)

Start-Sleep -Seconds 2
$child = Get-Process -Id $process.Id -ErrorAction SilentlyContinue
if ($null -eq $child) {
    $process.WaitForExit()
    [System.IO.File]::AppendAllText(
        $bootstrapLog,
        ("[{0}] Python-bootstrap PID {1} exited within bootstrap window with code {2}.{3}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $process.Id, $process.ExitCode, [Environment]::NewLine),
        [System.Text.Encoding]::UTF8
    )
}
else {
    [System.IO.File]::AppendAllText(
        $bootstrapLog,
        ("[{0}] Python-bootstrap PID {1} still alive after 2 seconds.{2}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $process.Id, [Environment]::NewLine),
        [System.Text.Encoding]::UTF8
    )
}

Write-Host "Started direct real-world training launcher."
Write-Host "Python-bootstrap PID: $($process.Id)"
Write-Host "Bootstrap log: $bootstrapLog"
Write-Host "Stdout log: $stdoutLog"
Write-Host "Stderr log: $stderrLog"
Write-Host "Connect timeout seconds: $ConnectTimeoutSeconds"
Write-Host "Dry run: $DryRun"
Write-Host "Window mode: $WindowMode"
Write-Host "Force CPU: $ForceCpu"
