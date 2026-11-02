param(
    [Parameter(Mandatory = $true)][string]$Executable,
    [string[]]$TestArguments = @(),
    [string]$Label = ''
)

$ErrorActionPreference = 'Stop'
$projectDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$outputDirectory = Join-Path $projectDirectory 'validation\live-tests'
$viewerPath = Join-Path $projectDirectory 'target\release\voxel-test-viewer.exe'
if (-not (Test-Path -LiteralPath $viewerPath)) {
    throw 'Build voxel-test-viewer before starting a visible test.'
}
[IO.Directory]::CreateDirectory($outputDirectory) | Out-Null
$env:VOXEL_LIVE_TESTS_DIR = $outputDirectory
if (-not $Label) { $Label = [IO.Path]::GetFileNameWithoutExtension($Executable) }
$env:VOXEL_LIVE_TEST_NAME = $Label
$env:TEMP = $outputDirectory
$env:TMP = $outputDirectory

$viewer = Get-Process -Name 'voxel-test-viewer' -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -eq $viewerPath } | Select-Object -First 1
if (-not $viewer) {
    Start-Process -FilePath $viewerPath -WorkingDirectory $projectDirectory `
        -ArgumentList @('--dir', ('"' + $outputDirectory + '"')) `
        -RedirectStandardOutput (Join-Path $outputDirectory 'viewer.stdout.log') `
        -RedirectStandardError (Join-Path $outputDirectory 'viewer.stderr.log') | Out-Null
}

$runId = [guid]::NewGuid().ToString('N')
$statusPath = Join-Path $outputDirectory 'run-status.json'
$temporaryStatus = Join-Path $outputDirectory ($runId + '.status.tmp')
$logPath = Join-Path $outputDirectory ($runId + '.log')
$status = [ordered]@{
    version = 1; run_id = $runId; label = $Label; status = 'running'
    passed = 0; failed = 0; ignored = 0; detail = 'Starting'
    started_utc = [DateTimeOffset]::UtcNow.ToString('O'); exit_code = $null
}
function Publish-RunStatus {
    [IO.File]::WriteAllText($temporaryStatus, ($status | ConvertTo-Json -Depth 4))
    Move-Item -LiteralPath $temporaryStatus -Destination $statusPath -Force
}
Publish-RunStatus

Push-Location -LiteralPath $projectDirectory
try {
    & $Executable @TestArguments 2>&1 | ForEach-Object {
        $line = $_.ToString()
        [IO.File]::AppendAllText($logPath, $line + [Environment]::NewLine)
        Write-Output $line
        $status.detail = $line
        if ($line -match '^test .+ \.\.\. (ok|FAILED|ignored)\s*$') {
            switch ($Matches[1]) {
                'ok' { $status.passed++ }
                'FAILED' { $status.failed++ }
                'ignored' { $status.ignored++ }
            }
        }
        Publish-RunStatus
    }
    $testExitCode = $LASTEXITCODE
    if (($status.passed + $status.failed + $status.ignored) -eq 0) {
        if ($testExitCode -eq 0) { $status.passed = 1 } else { $status.failed = 1 }
    }
    $status.exit_code = $testExitCode
    $status.status = if ($testExitCode -eq 0) { 'passed' } else { 'failed' }
    $status.finished_utc = [DateTimeOffset]::UtcNow.ToString('O')
    Publish-RunStatus
} catch {
    $status.status = 'failed'
    $status.detail = $_.Exception.Message
    $status.exit_code = 1
    Publish-RunStatus
    throw
} finally {
    Pop-Location
}
exit $testExitCode
