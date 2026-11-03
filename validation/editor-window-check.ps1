param(
    [Parameter(Mandatory=$true)][int]$TargetPid,
    [Parameter(Mandatory=$true)][string]$OutputPath,
    [int]$ClientWidth=0,
    [int]$ClientHeight=0,
    [switch]$SaveScene,
    [switch]$SelectFirst,
    [switch]$ToggleSimulation,
    [switch]$ToggleFullscreen
)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Drawing
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class EditorWindowCheck {
    [StructLayout(LayoutKind.Sequential)] public struct Rect { public int Left, Top, Right, Bottom; }
    [StructLayout(LayoutKind.Sequential)] public struct Point { public int X, Y; }
    [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
    [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr h, out Rect r);
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out Rect r);
    [DllImport("user32.dll")] public static extern bool ClientToScreen(IntPtr h, ref Point p);
    [DllImport("user32.dll")] public static extern bool MoveWindow(IntPtr h,int x,int y,int w,int z,bool redraw);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h,int command);
    [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr h);
    [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr h,IntPtr after,int x,int y,int w,int z,uint flags);
    [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr h,uint message,IntPtr w,IntPtr l);
    [DllImport("user32.dll")] public static extern uint MapVirtualKey(uint key,uint mode);
}
'@
[EditorWindowCheck]::SetProcessDPIAware() | Out-Null
$app=Get-Process -Id $TargetPid
$handle=$app.MainWindowHandle
if ($handle -eq 0) { throw 'Editor window is not ready.' }
if ([EditorWindowCheck]::IsIconic($handle)) { [EditorWindowCheck]::ShowWindow($handle,9) | Out-Null }
Start-Sleep -Milliseconds 250
$client=New-Object EditorWindowCheck+Rect
[EditorWindowCheck]::GetClientRect($handle,[ref]$client) | Out-Null
if ($client.Right -le 0 -or $client.Bottom -le 0) { throw 'Editor has no visible client area.' }
if ($ClientWidth -gt 0 -and $ClientHeight -gt 0) {
    [EditorWindowCheck]::ShowWindow($handle,9) | Out-Null
    $outer=New-Object EditorWindowCheck+Rect
    [EditorWindowCheck]::GetWindowRect($handle,[ref]$outer) | Out-Null
    $borderWidth=($outer.Right-$outer.Left)-$client.Right
    $borderHeight=($outer.Bottom-$outer.Top)-$client.Bottom
    [EditorWindowCheck]::MoveWindow($handle,0,0,$ClientWidth+$borderWidth,$ClientHeight+$borderHeight,$true) | Out-Null
    Start-Sleep -Milliseconds 400
    [EditorWindowCheck]::GetClientRect($handle,[ref]$client) | Out-Null
}
function Click-Canvas([double]$x,[double]$y) {
    $scale=[Math]::Min([Math]::Min($client.Bottom/780.0,$client.Right/960.0),2.5)
    $pixelX=[int][Math]::Round($x*$scale)
    $pixelY=[int][Math]::Round($y*$scale)
    $position=[IntPtr](($pixelY -shl 16) -bor ($pixelX -band 0xffff))
    [EditorWindowCheck]::PostMessage($handle,0x200,[IntPtr]::Zero,$position) | Out-Null
    [EditorWindowCheck]::PostMessage($handle,0x201,[IntPtr]1,$position) | Out-Null
    [EditorWindowCheck]::PostMessage($handle,0x202,[IntPtr]::Zero,$position) | Out-Null
    Start-Sleep -Milliseconds 200
}
if ($ToggleFullscreen) {
    $scan=[int64][EditorWindowCheck]::MapVirtualKey(0x7a,0)
    $down=[IntPtr](1 -bor ($scan -shl 16))
    $up=[IntPtr](1 -bor ($scan -shl 16) -bor 0xc0000000L)
    [EditorWindowCheck]::PostMessage($handle,0x100,[IntPtr]0x7a,$down) | Out-Null
    [EditorWindowCheck]::PostMessage($handle,0x101,[IntPtr]0x7a,$up) | Out-Null
    Start-Sleep -Milliseconds 350
    [EditorWindowCheck]::GetClientRect($handle,[ref]$client) | Out-Null
}
if ($SelectFirst) { Click-Canvas 50 178 }
if ($ToggleSimulation) { Click-Canvas 658 48 }
if ($SaveScene) { Click-Canvas 180 14; Click-Canvas 200 218 }
$point=New-Object EditorWindowCheck+Point
[EditorWindowCheck]::ClientToScreen($handle,[ref]$point) | Out-Null
[EditorWindowCheck]::SetWindowPos($handle,[IntPtr](-1),0,0,0,0,0x13) | Out-Null
Start-Sleep -Milliseconds 200
$bitmap=New-Object System.Drawing.Bitmap($client.Right,$client.Bottom)
$graphics=[System.Drawing.Graphics]::FromImage($bitmap)
$graphics.CopyFromScreen($point.X,$point.Y,0,0,$bitmap.Size)
$bitmap.Save($OutputPath,[System.Drawing.Imaging.ImageFormat]::Png)
$graphics.Dispose()
$bitmap.Dispose()
[EditorWindowCheck]::SetWindowPos($handle,[IntPtr](-2),0,0,0,0,0x13) | Out-Null
[pscustomobject]@{width=$client.Right;height=$client.Bottom;responding=$app.Responding;title=$app.MainWindowTitle;image=$OutputPath} | ConvertTo-Json -Compress
