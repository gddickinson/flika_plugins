run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Test: nms_detector ===
// Non-maximum suppression detector
print("Running test: nms_detector");
// Clear previous thunderSTORM results to prevent data carryover
// Close the ThunderSTORM results window if it exists
if (isOpen("ThunderSTORM: results")) {
    selectWindow("ThunderSTORM: results");
    run("Close");
}
wait(500);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/Endothelial_NonBapta_bin10_crop.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Non-maximum suppression] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/test_data/comparison_tests/imagej_results/nms_detector_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
// Save timing to file
f = File.open("/Users/george/claude_test/spt_batch_analysis/test_data/comparison_tests/imagej_results/nms_detector_imagej_timing.txt");
print(f, "nms_detector," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test nms_detector complete.");
