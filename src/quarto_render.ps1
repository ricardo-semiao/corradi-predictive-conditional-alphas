Copy-Item -Path README.md -Destination index.md
quarto render --no-clean --wrap=none
Remove-Item -Path index.md
Move-Item -Path "docs/distaso2026_empirical_analysis.pdf" -Destination "report.pdf" -Force
