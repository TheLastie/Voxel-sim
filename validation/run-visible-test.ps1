param(
    [Parameter(Mandatory = $true)][string]$ScenePath,
    [ValidateRange(2, 10000)][int]$Side = 16,
    [switch]$Create
)

# Compatibility filename: this opens an ordinary scene in the engine editor.
$ErrorActionPreference = 'Stop'
$projectDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$editorPath = Join-Path $projectDirectory 'target\release\voxel-editor.exe'
if (-not (Test-Path -LiteralPath $editorPath)) { throw 'Build voxel-editor before opening a scene.' }
$resolvedScenePath = [IO.Path]::GetFullPath($ScenePath)
$sceneArguments = if ($Create) {
    @('--create-impact-scene', ('"' + $resolvedScenePath + '"'), '--side', $Side.ToString())
} else {
    if (-not (Test-Path -LiteralPath $resolvedScenePath)) { throw 'Scene file does not exist.' }
    @('--scene', ('"' + $resolvedScenePath + '"'))
}
$outputDirectory = Join-Path $projectDirectory 'validation\live-tests'
[IO.Directory]::CreateDirectory($outputDirectory) | Out-Null
$env:TEMP = $outputDirectory
$env:TMP = $outputDirectory
Start-Process -FilePath $editorPath -WorkingDirectory $projectDirectory `
    -ArgumentList $sceneArguments `
    -RedirectStandardOutput (Join-Path $outputDirectory 'editor.stdout.log') `
    -RedirectStandardError (Join-Path $outputDirectory 'editor.stderr.log')
