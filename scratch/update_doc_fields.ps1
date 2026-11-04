param(
    [Parameter(Mandatory=$true)]
    [string]$FilePath
)

$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {
    $fullPath = (Resolve-Path $FilePath).Path
    Write-Output "Opening: $fullPath"
    $doc = $word.Documents.Open($fullPath)
    Write-Output "Document opened. Paragraphs: $($doc.Paragraphs.Count)"
    
    # Update fields in the document
    $doc.Fields.Update()
    
    # Also specifically update any TablesOfContents
    foreach ($toc in $doc.TablesOfContents) {
        $toc.Update()
    }
    
    Write-Output "All fields and TOC updated successfully!"
    $doc.Save()
    $doc.Close()
    Write-Output "Document saved and closed successfully."
}
catch {
    Write-Error "Error updating Word fields: $_"
}
finally {
    $word.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()
}
