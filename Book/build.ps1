# Build the SDP report.
#
#   powershell -ExecutionPolicy Bypass -File build.ps1
#
# Compiles inside .\build so that a PDF viewer holding sdp.pdf open cannot
# break the run, then copies the finished PDF out.  If the copy fails the
# build itself is still good and build\sdp.pdf is the current output.

$mik = "C:\Users\Salman\AppData\Local\Programs\MiKTeX\miktex\bin\x64"
$env:PATH = "$mik;$env:PATH"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force build | Out-Null

$tex = "$mik\pdflatex.exe"
$args = @("-interaction=nonstopmode", "-file-line-error",
          "-output-directory=build", "sdp.tex")

Write-Host "== pass 1 =="
& $tex $args | Out-Null
Write-Host "== bibtex =="
Copy-Item references.bib build\ -Force -ErrorAction SilentlyContinue
Push-Location build
& "$mik\bibtex.exe" sdp | Out-String | Write-Host
Pop-Location
Write-Host "== pass 2 =="
& $tex $args | Out-Null
Write-Host "== pass 3 =="
& $tex $args | Out-Null

Write-Host "`n===== ERRORS ====="
$err = Select-String -Path build\sdp.log -Pattern "^!|ignored error|Emergency stop" -ErrorAction SilentlyContinue
if ($err) { $err | Select-Object -First 20 | ForEach-Object { $_.Line } } else { Write-Host "none" }

Write-Host "`n===== UNDEFINED ====="
$und = Select-String -Path build\sdp.log -Pattern "Citation .* undefined|Reference .* undefined|There were undefined" -ErrorAction SilentlyContinue
if ($und) { $und | Select-Object -First 20 | ForEach-Object { $_.Line } } else { Write-Host "none" }

Write-Host "`n===== OVERFULL VBOX (content past page bottom) ====="
$ov = Select-String -Path build\sdp.log -Pattern "Overfull \\vbox" -ErrorAction SilentlyContinue
if ($ov) { $ov | Select-Object -First 10 | ForEach-Object { $_.Line } } else { Write-Host "none" }

Write-Host "`n===== RESULT ====="
Select-String -Path build\sdp.log -Pattern "Output written on" | ForEach-Object { $_.Line }

# Do NOT copy .aux/.toc/.lof/.lot/.bbl back to the root: pdflatex searches the
# current directory before the output directory, so a stale copy in the root
# shadows the fresh one in build\ and citations never resolve.
try {
    Copy-Item build\sdp.pdf .\sdp.pdf -Force -ErrorAction Stop
    Write-Host "copied -> sdp.pdf"
} catch {
    Write-Host "NOTE: sdp.pdf is open in a viewer and could not be replaced."
    Write-Host "      The current build is build\sdp.pdf"
}
