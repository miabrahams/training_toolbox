# Define the root directories
$rootPath = "D:\Sync_AI\Training\Bats\Images"
$imageRoot = "${rootPath}\Generated"
$maskRoot = "${rootPath}\Images\Masks"
$maskBuilder = "${rootPath}\MaskBuilder"

# Get all image files
$imageFiles = Get-ChildItem -Path $imageRoot -Recurse -Include *.png,*.jpg,*.jpeg

foreach ($imageFile in $imageFiles) {
    # Construct the expected mask path
    $relativePath = $imageFile.FullName.Substring($imageRoot.Length + 1)
    $expectedMaskPath = Join-Path $maskRoot $relativePath

    # Check if mask exists (ignoring extension)
    $maskFile = Get-ChildItem -Recurse -Path $maskRoot -Filter ($imageFile.BaseName + ".*") | Where-Object { $_.Extension -match '\.(png|jpg|jpeg)$' } | Select-Object -First 1

    if ($maskFile) {
        # If mask exists but in wrong location, move it
        if ($maskFile.FullName -ne $expectedMaskPath) {
            $targetDir = Split-Path $expectedMaskPath
            if (-not (Test-Path $targetDir)) {
                New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
            }
            Move-Item -Path $maskFile.FullName -Destination $expectedMaskPath -Force
            Write-Host "Moved mask: $($maskFile.FullName) -> $expectedMaskPath"
        }
    } else {
        # If mask doesn't exist, print to console and move image file to build mask
        Write-Host "No mask found for image: $($imageFile.FullName)"
        Copy-Item $imageFile $maskBuilder
    }
}