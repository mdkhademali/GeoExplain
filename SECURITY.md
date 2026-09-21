## Security Policy

## Supported versions

Only the latest release (currently 0.1.0, alpha) receives fixes.

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Use GitHub's private vulnerability reporting ("Security" tab → "Report a vulnerability") on https://github.com/mdkhademali/GeoExplain, or contact the maintainer through the contact options on the GitHub profile https://github.com/mdkhademali. Include a description, reproduction steps, affected version and (if possible) a suggested fix. You should receive an acknowledgement within about a week; this is a volunteer-maintained research project, so timelines are best-effort.

## Known security considerations

* **Model files are pickles.** `GeoExplainModel.save/load` use `joblib`, which can execute arbitrary code when loading. Only load model files from sources you trust.
* GeoExplain reads user-supplied CSV and GeoTIFF files with widely used libraries (pandas, rasterio); keep those dependencies up to date.
* The CLI does not execute shell commands and the package contains no credentials, tokens or network calls.
