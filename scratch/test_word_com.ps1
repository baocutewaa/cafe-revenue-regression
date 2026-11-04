$word = New-Object -ComObject Word.Application
$word.Visible = $false
Write-Output "Word version: $($word.Version)"
$docPath = (Resolve-Path "scratch\test_full_toc.docx").Path
$doc = $word.Documents.Open($docPath)
Write-Output "Document opened successfully! Paragraph count: $($doc.Paragraphs.Count)"
$doc.Fields.Update()
Write-Output "Fields updated successfully!"
$doc.Save()
$doc.Close()
$word.Quit()
Write-Output "Done!"
