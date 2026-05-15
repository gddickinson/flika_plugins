# LocsAndTracksPlotter — Interface Map

Top-level navigation guide for this FLIKA plugin. Read this before opening
individual modules.

## Architecture overview

```
flika main window
   │
   └── LocsAndTracksPlotter (locsAndTracksPlotter.py — main entry, plugin instance)
        │
        ├── TrackPlotOptions   – point/track display, intensity choice, BG subtract
        ├── FilterOptions      – numeric filtering on data columns
        ├── TrackWindow        – per-track plots (intensity, position, NN, velocity, …)
        ├── TrackPlot          – single-track 2D image + line overlay
        ├── AllTracksPlot      – overview plot of every track
        ├── FlowerPlotWindow   – tracks plotted from a common origin
        ├── ROIPLOT            – ROI-restricted track view
        ├── Overlay            – image-overlay configuration
        ├── ChartDock          – scatter / line / histogram tool
        ├── DiffusionPlotWindow – MSD / diffusion analysis
        └── JoinTracks         – merge fragmented tracks, recompute metrics
```

## Files

| File | Purpose | Key classes / functions |
| --- | --- | --- |
| `__init__.py` | Loads PyQtGraph compatibility patches, imports plugin instance. | `setPoints` (compat), `setBackground` (compat) |
| `locsAndTracksPlotter.py` | Main plugin class, GUI wiring, filtering, track update flow. | `LocsAndTracksPlotter`, `TrackPlotOptions`, `FilterOptions`, `dictFromList` |
| `trackWindow.py` | Per-track analysis window — labels (SVM, Length, sRg), intensity, position, NN, velocity, variance plots. | `TrackWindow`, `_update_info_labels`, `update_column_selector` |
| `trackPlotter.py` | Single-track image + plot, ROI cropping. | `TrackPlot`, `backgroundSubtractStack` |
| `allTracksPlotter.py` | All-tracks overview. | `AllTracksPlot` |
| `chartDock.py` | Dockable scatter/line/histogram chart. | `ChartDock` |
| `diffusionPlot.py` | MSD plotting, diffusion coefficient estimation. | `DiffusionPlotWindow` |
| `flowerPlot.py` | Tracks rendered from common origin. | `FlowerPlotWindow` |
| `roiZoomPlotter.py` | ROI-zoom track inspector. | `ROIPLOT` |
| `overlay.py` | Image overlay configuration. | `Overlay` |
| `joinTracks.py` | Track joining + recomputation of radius-of-gyration metrics. | `JoinTracks` |
| `io.py` | CSV / JSON I/O helpers. | (loaders) |
| `helperFunctions.py` | Shared utilities (`rollingFunc`, `dictFromList`, etc.). | |
| `scaleBarGUI.py` | Scale-bar configuration GUI. | |
| `pyqtgraphWindowRecorder.py` / `screenRecorder.py` | Screen capture helpers. | |
| `standalone_app.py` | Run without flika for development. | |
| `combineImages.py`, `diagnose_tracks.py`, `diagnostic_script.py`, `trackpyTest.py`, `localVglobalThresholding_skimageExample.py`, `example_usage.py` | Auxiliary / example scripts. | |

## Track-update data flow

1. User presses `T` over a point → `LocsAndTracksPlotter.selectTrack` sets
   `displayTrack` and calls `_update_track_displays`.
2. `_update_track_displays` reads the current intensity column from
   `TrackPlotOptions.intensityChoice_Box`, optionally subtracts the
   background value, gathers per-track metrics (`_extract_track_metrics`),
   and calls `TrackWindow.update(...)`.
3. `TrackWindow.update` populates each plot and refreshes the info labels
   (track id, SVM, length, sRg) and the user-selected column value.

## Filter data flow

1. `FilterOptions` widgets (column / operator / value) drive
   `LocsAndTracksPlotter.filterData`.
2. The result is stored in `self.filteredData` and `useFilteredData=True`;
   `plotPointData` and downstream views inspect those flags.

## Companion plugin

The CSV files this plugin reads are produced by
`/Users/george/claude_test/spt_batch_analysis`. See its `ROADMAP.md` and
this plugin's `ROADMAP.md` for known integration issues.

## Conventions

- All scripts should stay under 500 lines where practical; split into
  focused modules first.
- Update this file whenever the module layout changes.
