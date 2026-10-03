# Publish to GitHub

## 1. Create the repository

Create an empty GitHub repository named `LVIMOT` without auto-generating a README, `.gitignore` or license.

## 2. Push this repository

From the extracted `LVIMOT_GitHub_Ready` directory:

```bash
git init
git add .
git commit -m "Initial LVIMOT release"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/LVIMOT.git
git push -u origin main
```

## 3. Publish the Windows release

Create a GitHub release with:

- Tag: `windows-v1.0.0`
- Title: `LVIMOT Windows v1.0.0`
- Asset: `LVIMOT-Windows-v1.0.0.zip`
- Notes: use `RELEASE_NOTES_WINDOWS.md`

## 4. Publish the Linux release

Create a second GitHub release with:

- Tag: `linux-v1.0.0`
- Title: `LVIMOT Linux v1.0.0`
- Asset: `LVIMOT-Linux-v1.0.0.zip`
- Notes: use `RELEASE_NOTES_LINUX.md`

The repository intentionally excludes CARLA, datasets, virtual environments, generated outputs, model weights and machine-specific GUI configuration.
