$ErrorActionPreference = 'Stop'
$python = 'C:\Users\牧之原\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$script = '\\wsl.localhost\Ubuntu-22.04\home\shiqiany\AIagent\PhomatAgent\tools\render_pdf_pages.py'
$pdf = '\\wsl.localhost\Ubuntu-22.04\tmp\crystalagent_q1_render.pdf'
$pages = '\\wsl.localhost\Ubuntu-22.04\tmp\crystalagent_pages'
& $python $script $pdf $pages
if ($LASTEXITCODE -ne 0) { throw "PDF render failed with exit code $LASTEXITCODE" }
