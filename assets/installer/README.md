# Installer artwork

`background.svg` is the editable 680 × 440 point source. `background.tiff` is the
Finder-ready asset with 1× and 2× representations (680 × 440 at 72 dpi and
1360 × 880 at 144 dpi), combined using macOS `tiffutil -cathidpicheck`.
Render the SVG with Didot and the system sans font available before exporting.
The checked-in TIFF means packaging needs no browser or rasterization dependency.

Keep icon centers at (170, 228) and (510, 228), matching
`installerLayoutScript` in `xerxes/scripts/buildDesktopInstaller.ts`. Finder owns
the two actual icons and their labels; the background contains no fake controls.
Use a light background under Finder's black icon labels.
