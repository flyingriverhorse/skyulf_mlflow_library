<#
Set repository About information and topics using GitHub CLI (gh).

Usage (PowerShell):
  ./scripts/set_github_about.ps1 -Owner flyingriverhorse -Repo skyulf_mlflow_library

Requirements:
  - GitHub CLI (gh) installed and authenticated (gh auth login)
  - Your gh session must have repo permissions
#>

param(
  [string]$Owner = "flyingriverhorse",
  [string]$Repo = "skyulf_mlflow_library",
  [string]$Description = "Lightweight ML preprocessing, feature engineering and model registry library.",
  [string]$Homepage = "https://github.com/flyingriverhorse/skyulf_mlflow_library",
  [string[]]$Topics = @(
    "ml",
    "mlflow",
    "feature-engineering",
    "data-quality",
    "preprocessing",
    "model-registry",
    "python",
    "machine-learning"
  )
)

Write-Host "Setting repository description and homepage..."
gh repo edit "${Owner}/${Repo}" --description "$Description" --homepage "$Homepage"

Write-Host "Setting repository topics..."
# Use the Topics API via gh api because gh repo edit does not reliably set topics
$body = @{ names = $Topics } | ConvertTo-Json
gh api --method PUT -H "Accept: application/vnd.github+json" /repos/${Owner}/${Repo}/topics -f names="$($Topics -join ',')"

Write-Host "Done. Verify the About section on GitHub: https://github.com/$Owner/$Repo"
