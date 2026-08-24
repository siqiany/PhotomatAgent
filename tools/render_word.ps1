$ErrorActionPreference = 'Stop'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {
    $inputPath = '\\wsl.localhost\Ubuntu-22.04\home\shiqiany\AIagent\PhomatAgent\CrystalAgent_Q1_revised.docx'
    $outputPath = '\\wsl.localhost\Ubuntu-22.04\tmp\crystalagent_q1_render.pdf'
    $doc = $word.Documents.Open($inputPath)
    try {
        $doc.ExportAsFixedFormat($outputPath, 17)
    }
    finally {
        $doc.Close()
    }
}
finally {
    $word.Quit()
}
Write-Output $outputPath
