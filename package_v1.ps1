# Simple PowerShell Script to Create a Clean Deployment Zip File
$ZipFile = "ECG_Arrhythmia_Service_v1.zip"
$ProjectRoot = "c:\Users\admin\Documents\porject\Project_Submission_Clean"
$StagingDir = Join-Path $ProjectRoot "staging_v1"

# Essential Files
$RootFiles = "kafka_consumer.py", "ecg_processor.py", "mongo_writer.py", "config.py", "Dockerfile", ".dockerignore", "requirements.txt", ".env.template", "test_producer.py", "docker-compose.yml"
# Essential Directories
$Directories = "signal_processing", "models_training", "decision_engine", "xai", "data", "utils"

if (Test-Path $ZipFile) { Remove-Item $ZipFile }
if (Test-Path $StagingDir) { Remove-Item -Recurse -Force $StagingDir }
New-Item -ItemType Directory -Path $StagingDir | Out-Null

# Simple file-by-file copy for root files
foreach ($file in $RootFiles) {
    if (Test-Path -Path (Join-Path $ProjectRoot $file)) {
        Copy-Item (Join-Path $ProjectRoot $file) $StagingDir
    }
}

# Recursively copy directories using robocopy (more reliable on Windows)
foreach ($dir in $Directories) {
    $src = Join-Path $ProjectRoot $dir
    $dst = Join-Path $StagingDir $dir
    if (Test-Path $src) {
        # Robocopy for speed and path-depth handling
        # /E: Copy subdirectories /XD: Exclude directories /XF: Exclude files
        robocopy $src $dst /E /XD "__pycache__" ".git" "venv" ".vscode" /XF "*.pyc" /NFL /NDL /NJH /NJS /nc /ns /np
    }
}

Write-Host "Compressing Archive..." -ForegroundColor Cyan
Compress-Archive -Path "$StagingDir\*" -DestinationPath $ZipFile

# Cleanup
Remove-Item -Recurse -Force $StagingDir
Write-Host "✅ Zip Created Successfully: $ZipFile" -ForegroundColor Green
