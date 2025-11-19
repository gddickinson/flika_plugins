
GlobalAnalysisPlugin
====================

Global Analysis plugin for FLIKA program. Fit user-drawn regions of interest to polynomials for rise-fall time calculation.

.. image:: GlobalAnalysisSample.gif

Open the Global Analysis UI from the plugins menu, and plot the ROIs in the window you would like to analyze.

Check the "Show Trace ROI" Checkbox to display an ROI within the trace window.  The region within that range will be 
fitted to a polynomial and used to calculate rise and fall times.

Use the ROI selector to choose which ROI trace to fit.
The Puff Selector drop down breifly scans the trace for peaks.

Use the region of interest to select the puff to fit:

- Slide the left and right boundaries to select frames
- Use the bottom range to set the baseline value.
- The fitted region will be plotted over the trace, with Rise/Fall times listed in the table
- Numpy poly1d function is used to fit the image, and the 20%, 50%, 80% values are calculated compared to the baseline
- Use log data to add values to a text file, and save when complete
