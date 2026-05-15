# LocsAndTracksPlotter — Roadmap

Plan and status for outstanding bug fixes / improvements requested by users.

## In Progress (2026-05-15 — current development cycle)

### Track Window UI

- [x] **sRg (scaled radius of gyration) missing from track info window.**
  The label only shows `SVM = X, Length = Y`. Extend it to also show the
  available sRg column (`sRg_geometric` or `radius_gyration_scaled`).
  *File:* `trackWindow.py` (`_update_info_labels`).

- [x] **Default column selector falls back too aggressively.**
  When `sRg_geometric` is not present, the selector defaults to the first
  column (typically `frame`). Prefer `radius_gyration_scaled` as a
  secondary default so users can see the per-track sRg value immediately.
  *File:* `trackWindow.py` (`update_column_selector`).

- [x] **Duplicate `_update_info_labels` method.** The first definition wires
  up the custom column value display; the second overrides it without that
  call. Remove the duplicate so the column value updates without explicit
  refresh.

### Track Filter

- [x] **Filter on `radius_gyration_scaled` (and other track-level metrics)
  silently produces no results.**
  If a column has all-NaN values, NaN comparisons return False so every row
  is filtered out and the user just sees an empty plot. Add an explicit
  NaN check and warning. Also surface the result count even when 0.
  *File:* `locsAndTracksPlotter.py` (`filterData`).

### Intensity Plot

- [x] **Background-subtracted intensity cannot be displayed.**
  - The intensity-choice ComboBox, Background-Subtract checkbox, and
    background value spinbox don't have change handlers, so toggling them
    doesn't refresh the track window. Users must press `T` again.
  - If a non-existent or all-NaN intensity column is chosen, the plot
    silently fails. Fall back to `intensity` and log a status message.
  *Files:* `locsAndTracksPlotter.py` (`TrackPlotOptions._create_background_options`,
  `_update_track_displays`).

## Planned

### Companion plugin (`spt_batch_analysis`)

- [x] **Detection method (Thunderstorm vs. utrack) does not persist on
  load.** `update_gui_from_parameters` doesn't set
  `utrack_method_radio` / `thunderstorm_method_radio` from the loaded
  `parameters.detection_method`. Add the missing radio-button updates and
  call `on_detection_method_changed` afterwards.
  *File:* `spt_batch_analysis/__init__.py` (`update_gui_from_parameters`).

- [x] **Thunderstorm macro generator tab parameters are not persisted.**
  Widgets in that tab (`ts_filter_type`, `ts_wavelet_scale`, `ts_detector`,
  `ts_estimator`, `ts_sigma`, MFA options, renderer, export columns,
  PyImageJ options, etc.) are independent from the Detection tab's
  Thunderstorm widgets and are never read into / written from
  `SPTAnalysisParameters`. Add `ts_macro_*` fields and wire them through
  `update_parameters` and `update_gui_from_parameters`.
  *File:* `spt_batch_analysis/__init__.py`.

## Done

(see git history)

## Future / Nice to Have

- Validate intensity-choice dropdown options against actual data columns so
  the dropdown only shows the columns present in the loaded CSV.
- Consider per-track (vs. per-point) filtering mode for track-level metrics.
- Make the trackWindow column-selector remember its previous selection
  across data reloads.
