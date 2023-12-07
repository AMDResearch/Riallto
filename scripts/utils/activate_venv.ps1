$username = [System.Environment]::UserName
$venvPath = "C:\Users\$username\AppData\Local\Riallto\riallto_venv"

try {
    . "$venvPath\Scripts\activate.ps1"
} catch {
    Write-Host "Failed to enter riallto_venv virtual environment"
    Exit 1
}