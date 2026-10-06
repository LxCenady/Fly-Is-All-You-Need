# Rebuilds ../UCTF_Readout.exe from ../index.html with the C# compiler that ships with Windows (.NET Framework 4).
$csc = "$env:WINDIR\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
Set-Location $PSScriptRoot
& $csc /nologo /target:winexe /optimize+ /out:..\UCTF_Readout.exe /win32icon:gpf.ico /resource:..\index.html,index.html Launcher.cs
