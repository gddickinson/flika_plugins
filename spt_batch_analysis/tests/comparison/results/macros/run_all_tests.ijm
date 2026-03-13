// Auto-generated thunderSTORM comparison macros
// Input: /Users/george/claude_test/spt_batch_analysis/test_data/Endothelial_NonBapta_bin10_crop.tif
// Generated: 2026-03-12 16:09:49

// Set camera parameters
run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Test: wavelet_default ===
// Wavelet B-Spline filter, default settings
print("Running test: wavelet_default");
// Clear previous thunderSTORM results to prevent data carryover
// Close the ThunderSTORM results window if it exists
if (isOpen("ThunderSTORM: results")) {
    selectWindow("ThunderSTORM: results");
    run("Close");
}
wait(500);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/Endothelial_NonBapta_bin10_crop.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=4-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/test_data/comparison_tests/imagej_results/wavelet_default_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
// Save timing to file
f = File.open("/Users/george/claude_test/spt_batch_analysis/test_data/comparison_tests/imagej_results/wavelet_default_imagej_timing.txt");
print(f, "wavelet_default," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test wavelet_default complete.");

// === Test: wavelet_scale4_order5 ===
// Wavelet filter with scale=4, order=5
print("Running test: wavelet_scale4_order5");
// Clear previous thunderSTORM results to prevent data carryover
// Close the ThunderSTORM results window if it exists
if (isOpen("ThunderSTORM: results")) {
    selectWindow("ThunderSTORM: results");
    run("Close");
}
wait(500);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/Endothelial_NonBapta_bin10_crop.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=4.0 order=5 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/test_data/comparison_tests/imagej_results/wavelet_scale4_order5_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
// Save timing to file
f = File.open("/Users/george/claude_test/spt_batch_analysis/test_data/comparison_tests/imagej_results/wavelet_scale4_order5_imagej_timing.txt");
print(f, "wavelet_scale4_order5," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test wavelet_scale4_order5 complete.");

// === Test: dog_filter ===
// Difference of Gaussians filter
print("Running test: dog_filter");
// Clear previous thunderSTORM results to prevent data carryover
// Close the ThunderSTORM results window if it exists
if (isOpen("ThunderSTORM: results")) {
    selectWindow("ThunderSTORM: results");
    run("Close");
}
wait(500);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/Endothelial_NonBapta_bin10_crop.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Difference-of-Gaussians filter] sigma1=1.0 sigma2=1.6 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/test_data/comparison_tests/imagej_results/dog_filter_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
// Save timing to file
f = File.open("/Users/george/claude_test/spt_batch_analysis/test_data/comparison_tests/imagej_results/dog_filter_imagej_timing.txt");
print(f, "dog_filter," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test dog_filter complete.");

// === Test: gaussian_filter ===
// Gaussian (lowered) filter
print("Running test: gaussian_filter");
// Clear previous thunderSTORM results to prevent data carryover
// Close the ThunderSTORM results window if it exists
if (isOpen("ThunderSTORM: results")) {
    selectWindow("ThunderSTORM: results");
    run("Close");
}
wait(500);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/Endothelial_NonBapta_bin10_crop.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Lowered Gaussian filter] sigma=1.6 detector=[Local maximum] connectivity=8-neighbourhood threshold=1.5*std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/test_data/comparison_tests/imagej_results/gaussian_filter_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
// Save timing to file
f = File.open("/Users/george/claude_test/spt_batch_analysis/test_data/comparison_tests/imagej_results/gaussian_filter_imagej_timing.txt");
print(f, "gaussian_filter," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test gaussian_filter complete.");

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

// === Test: centroid_detector ===
// Centroid of connected components detector
print("Running test: centroid_detector");
// Clear previous thunderSTORM results to prevent data carryover
// Close the ThunderSTORM results window if it exists
if (isOpen("ThunderSTORM: results")) {
    selectWindow("ThunderSTORM: results");
    run("Close");
}
wait(500);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/Endothelial_NonBapta_bin10_crop.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Centroid of connected components] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/test_data/comparison_tests/imagej_results/centroid_detector_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
// Save timing to file
f = File.open("/Users/george/claude_test/spt_batch_analysis/test_data/comparison_tests/imagej_results/centroid_detector_imagej_timing.txt");
print(f, "centroid_detector," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test centroid_detector complete.");

// === Test: lsq_fitting ===
// Least squares fitting
print("Running test: lsq_fitting");
// Clear previous thunderSTORM results to prevent data carryover
// Close the ThunderSTORM results window if it exists
if (isOpen("ThunderSTORM: results")) {
    selectWindow("ThunderSTORM: results");
    run("Close");
}
wait(500);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/Endothelial_NonBapta_bin10_crop.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/test_data/comparison_tests/imagej_results/lsq_fitting_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
// Save timing to file
f = File.open("/Users/george/claude_test/spt_batch_analysis/test_data/comparison_tests/imagej_results/lsq_fitting_imagej_timing.txt");
print(f, "lsq_fitting," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test lsq_fitting complete.");

// === Test: mle_fitting ===
// Maximum likelihood estimation fitting
print("Running test: mle_fitting");
// Clear previous thunderSTORM results to prevent data carryover
// Close the ThunderSTORM results window if it exists
if (isOpen("ThunderSTORM: results")) {
    selectWindow("ThunderSTORM: results");
    run("Close");
}
wait(500);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/Endothelial_NonBapta_bin10_crop.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Maximum likelihood] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/test_data/comparison_tests/imagej_results/mle_fitting_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
// Save timing to file
f = File.open("/Users/george/claude_test/spt_batch_analysis/test_data/comparison_tests/imagej_results/mle_fitting_imagej_timing.txt");
print(f, "mle_fitting," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test mle_fitting complete.");

// === Test: psf_gaussian ===
// PSF: Gaussian (non-integrated) with WLSQ
print("Running test: psf_gaussian");
// Clear previous thunderSTORM results to prevent data carryover
// Close the ThunderSTORM results window if it exists
if (isOpen("ThunderSTORM: results")) {
    selectWindow("ThunderSTORM: results");
    run("Close");
}
wait(500);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/Endothelial_NonBapta_bin10_crop.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/test_data/comparison_tests/imagej_results/psf_gaussian_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
// Save timing to file
f = File.open("/Users/george/claude_test/spt_batch_analysis/test_data/comparison_tests/imagej_results/psf_gaussian_imagej_timing.txt");
print(f, "psf_gaussian," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test psf_gaussian complete.");

// === Test: radial_symmetry ===
// Radial symmetry estimator
print("Running test: radial_symmetry");
// Clear previous thunderSTORM results to prevent data carryover
// Close the ThunderSTORM results window if it exists
if (isOpen("ThunderSTORM: results")) {
    selectWindow("ThunderSTORM: results");
    run("Close");
}
wait(500);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/Endothelial_NonBapta_bin10_crop.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[Radial symmetry] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/test_data/comparison_tests/imagej_results/radial_symmetry_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
// Save timing to file
f = File.open("/Users/george/claude_test/spt_batch_analysis/test_data/comparison_tests/imagej_results/radial_symmetry_imagej_timing.txt");
print(f, "radial_symmetry," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test radial_symmetry complete.");

// === Test: mfa_enabled ===
// Multi-emitter fitting analysis enabled
print("Running test: mfa_enabled");
// Clear previous thunderSTORM results to prevent data carryover
// Close the ThunderSTORM results window if it exists
if (isOpen("ThunderSTORM: results")) {
    selectWindow("ThunderSTORM: results");
    run("Close");
}
wait(500);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/Endothelial_NonBapta_bin10_crop.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=true keep_same_intensity=true nmax=5 fixed_intensity=false expected_intensity=500:2500 pvalue=1.0E-6 renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/test_data/comparison_tests/imagej_results/mfa_enabled_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
// Save timing to file
f = File.open("/Users/george/claude_test/spt_batch_analysis/test_data/comparison_tests/imagej_results/mfa_enabled_imagej_timing.txt");
print(f, "mfa_enabled," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test mfa_enabled complete.");

// === Test: high_threshold ===
// Higher detection threshold (2x std)
print("Running test: high_threshold");
// Clear previous thunderSTORM results to prevent data carryover
// Close the ThunderSTORM results window if it exists
if (isOpen("ThunderSTORM: results")) {
    selectWindow("ThunderSTORM: results");
    run("Close");
}
wait(500);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/Endothelial_NonBapta_bin10_crop.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=2*std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/test_data/comparison_tests/imagej_results/high_threshold_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
// Save timing to file
f = File.open("/Users/george/claude_test/spt_batch_analysis/test_data/comparison_tests/imagej_results/high_threshold_imagej_timing.txt");
print(f, "high_threshold," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test high_threshold complete.");

// === Test: fitradius_5 ===
// Larger fit radius (5 pixels)
print("Running test: fitradius_5");
// Clear previous thunderSTORM results to prevent data carryover
// Close the ThunderSTORM results window if it exists
if (isOpen("ThunderSTORM: results")) {
    selectWindow("ThunderSTORM: results");
    run("Close");
}
wait(500);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/Endothelial_NonBapta_bin10_crop.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=5 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/test_data/comparison_tests/imagej_results/fitradius_5_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
// Save timing to file
f = File.open("/Users/george/claude_test/spt_batch_analysis/test_data/comparison_tests/imagej_results/fitradius_5_imagej_timing.txt");
print(f, "fitradius_5," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test fitradius_5 complete.");


print("All tests complete!");
