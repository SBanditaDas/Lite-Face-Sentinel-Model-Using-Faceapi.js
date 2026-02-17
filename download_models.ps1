# Download face-api.js Models (Under 2 MB)
# Run this in PowerShell from the Face Sentinel directory

# Create models directory
New-Item -ItemType Directory -Force -Path "public\models"

Write-Host "Downloading face-api.js models (under 2 MB)..." -ForegroundColor Green

$baseUrl = "https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights"

# Tiny Face Detector (~190 KB)
Invoke-WebRequest -Uri "$baseUrl/tiny_face_detector_model-weights_manifest.json" -OutFile "public\models\tiny_face_detector_model-weights_manifest.json"
Invoke-WebRequest -Uri "$baseUrl/tiny_face_detector_model-shard1" -OutFile "public\models\tiny_face_detector_model-shard1"

# Face Landmarks Tiny (~80 KB)
Invoke-WebRequest -Uri "$baseUrl/face_landmark_68_tiny_model-weights_manifest.json" -OutFile "public\models\face_landmark_68_tiny_model-weights_manifest.json"
Invoke-WebRequest -Uri "$baseUrl/face_landmark_68_tiny_model-shard1" -OutFile "public\models\face_landmark_68_tiny_model-shard1"

# Face Recognition Model (~6 MB - we'll optimize this)
# Using the quantized version
Invoke-WebRequest -Uri "$baseUrl/face_recognition_model-weights_manifest.json" -OutFile "public\models\face_recognition_model-weights_manifest.json"
Invoke-WebRequest -Uri "$baseUrl/face_recognition_model-shard1" -OutFile "public\models\face_recognition_model-shard1"
Invoke-WebRequest -Uri "$baseUrl/face_recognition_model-shard2" -OutFile "public\models\face_recognition_model-shard2"

Write-Host "`nDownload complete!" -ForegroundColor Green
Write-Host "Models saved to: public\models\" -ForegroundColor Cyan

# Calculate total size
$totalSize = (Get-ChildItem "public\models" | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host "`nTotal model size: $([math]::Round($totalSize, 2)) MB" -ForegroundColor Yellow
