run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: sparse_108nm__wavelet_default ===
// Dataset: Sparse molecules, 108nm pixel (match real data)  Algorithm: Wavelet B-Spline filter, default settings
print("Running synthetic test: sparse_108nm__wavelet_default");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=4-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_108nm__wavelet_default_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_108nm__wavelet_default_imagej_timing.txt");
print(f, "sparse_108nm__wavelet_default," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_108nm__wavelet_default complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: sparse_108nm__wavelet_scale4_order5 ===
// Dataset: Sparse molecules, 108nm pixel (match real data)  Algorithm: Wavelet filter with scale=4, order=5
print("Running synthetic test: sparse_108nm__wavelet_scale4_order5");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=4.0 order=5 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_108nm__wavelet_scale4_order5_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_108nm__wavelet_scale4_order5_imagej_timing.txt");
print(f, "sparse_108nm__wavelet_scale4_order5," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_108nm__wavelet_scale4_order5 complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: sparse_108nm__dog_filter ===
// Dataset: Sparse molecules, 108nm pixel (match real data)  Algorithm: Difference of Gaussians filter
print("Running synthetic test: sparse_108nm__dog_filter");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Difference-of-Gaussians filter] sigma1=1.0 sigma2=1.6 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_108nm__dog_filter_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_108nm__dog_filter_imagej_timing.txt");
print(f, "sparse_108nm__dog_filter," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_108nm__dog_filter complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: sparse_108nm__gaussian_filter ===
// Dataset: Sparse molecules, 108nm pixel (match real data)  Algorithm: Gaussian (lowered) filter
print("Running synthetic test: sparse_108nm__gaussian_filter");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Lowered Gaussian filter] sigma=1.6 detector=[Local maximum] connectivity=8-neighbourhood threshold=1.5*std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_108nm__gaussian_filter_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_108nm__gaussian_filter_imagej_timing.txt");
print(f, "sparse_108nm__gaussian_filter," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_108nm__gaussian_filter complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: sparse_108nm__nms_detector ===
// Dataset: Sparse molecules, 108nm pixel (match real data)  Algorithm: Non-maximum suppression detector
print("Running synthetic test: sparse_108nm__nms_detector");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Non-maximum suppression] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_108nm__nms_detector_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_108nm__nms_detector_imagej_timing.txt");
print(f, "sparse_108nm__nms_detector," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_108nm__nms_detector complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: sparse_108nm__centroid_detector ===
// Dataset: Sparse molecules, 108nm pixel (match real data)  Algorithm: Centroid of connected components detector
print("Running synthetic test: sparse_108nm__centroid_detector");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Centroid of connected components] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_108nm__centroid_detector_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_108nm__centroid_detector_imagej_timing.txt");
print(f, "sparse_108nm__centroid_detector," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_108nm__centroid_detector complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: sparse_108nm__lsq_fitting ===
// Dataset: Sparse molecules, 108nm pixel (match real data)  Algorithm: Least squares fitting
print("Running synthetic test: sparse_108nm__lsq_fitting");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_108nm__lsq_fitting_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_108nm__lsq_fitting_imagej_timing.txt");
print(f, "sparse_108nm__lsq_fitting," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_108nm__lsq_fitting complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: sparse_108nm__mle_fitting ===
// Dataset: Sparse molecules, 108nm pixel (match real data)  Algorithm: Maximum likelihood estimation fitting
print("Running synthetic test: sparse_108nm__mle_fitting");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Maximum likelihood] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_108nm__mle_fitting_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_108nm__mle_fitting_imagej_timing.txt");
print(f, "sparse_108nm__mle_fitting," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_108nm__mle_fitting complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: sparse_108nm__psf_gaussian ===
// Dataset: Sparse molecules, 108nm pixel (match real data)  Algorithm: PSF: Gaussian (non-integrated) with WLSQ
print("Running synthetic test: sparse_108nm__psf_gaussian");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_108nm__psf_gaussian_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_108nm__psf_gaussian_imagej_timing.txt");
print(f, "sparse_108nm__psf_gaussian," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_108nm__psf_gaussian complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: sparse_108nm__radial_symmetry ===
// Dataset: Sparse molecules, 108nm pixel (match real data)  Algorithm: Radial symmetry estimator
print("Running synthetic test: sparse_108nm__radial_symmetry");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[Radial symmetry] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_108nm__radial_symmetry_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_108nm__radial_symmetry_imagej_timing.txt");
print(f, "sparse_108nm__radial_symmetry," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_108nm__radial_symmetry complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: sparse_108nm__mfa_enabled ===
// Dataset: Sparse molecules, 108nm pixel (match real data)  Algorithm: Multi-emitter fitting analysis enabled
print("Running synthetic test: sparse_108nm__mfa_enabled");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=true keep_same_intensity=true nmax=5 fixed_intensity=false expected_intensity=500:2500 pvalue=1.0E-6 renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_108nm__mfa_enabled_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_108nm__mfa_enabled_imagej_timing.txt");
print(f, "sparse_108nm__mfa_enabled," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_108nm__mfa_enabled complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: sparse_108nm__high_threshold ===
// Dataset: Sparse molecules, 108nm pixel (match real data)  Algorithm: Higher detection threshold (2x std)
print("Running synthetic test: sparse_108nm__high_threshold");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=2*std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_108nm__high_threshold_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_108nm__high_threshold_imagej_timing.txt");
print(f, "sparse_108nm__high_threshold," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_108nm__high_threshold complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: sparse_108nm__fitradius_5 ===
// Dataset: Sparse molecules, 108nm pixel (match real data)  Algorithm: Larger fit radius (5 pixels)
print("Running synthetic test: sparse_108nm__fitradius_5");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=5 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_108nm__fitradius_5_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_108nm__fitradius_5_imagej_timing.txt");
print(f, "sparse_108nm__fitradius_5," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_108nm__fitradius_5 complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: medium_108nm__wavelet_default ===
// Dataset: Medium density, 108nm pixel  Algorithm: Wavelet B-Spline filter, default settings
print("Running synthetic test: medium_108nm__wavelet_default");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/medium_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=4-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_108nm__wavelet_default_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_108nm__wavelet_default_imagej_timing.txt");
print(f, "medium_108nm__wavelet_default," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test medium_108nm__wavelet_default complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: medium_108nm__wavelet_scale4_order5 ===
// Dataset: Medium density, 108nm pixel  Algorithm: Wavelet filter with scale=4, order=5
print("Running synthetic test: medium_108nm__wavelet_scale4_order5");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/medium_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=4.0 order=5 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_108nm__wavelet_scale4_order5_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_108nm__wavelet_scale4_order5_imagej_timing.txt");
print(f, "medium_108nm__wavelet_scale4_order5," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test medium_108nm__wavelet_scale4_order5 complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: medium_108nm__dog_filter ===
// Dataset: Medium density, 108nm pixel  Algorithm: Difference of Gaussians filter
print("Running synthetic test: medium_108nm__dog_filter");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/medium_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Difference-of-Gaussians filter] sigma1=1.0 sigma2=1.6 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_108nm__dog_filter_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_108nm__dog_filter_imagej_timing.txt");
print(f, "medium_108nm__dog_filter," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test medium_108nm__dog_filter complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: medium_108nm__gaussian_filter ===
// Dataset: Medium density, 108nm pixel  Algorithm: Gaussian (lowered) filter
print("Running synthetic test: medium_108nm__gaussian_filter");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/medium_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Lowered Gaussian filter] sigma=1.6 detector=[Local maximum] connectivity=8-neighbourhood threshold=1.5*std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_108nm__gaussian_filter_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_108nm__gaussian_filter_imagej_timing.txt");
print(f, "medium_108nm__gaussian_filter," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test medium_108nm__gaussian_filter complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: medium_108nm__nms_detector ===
// Dataset: Medium density, 108nm pixel  Algorithm: Non-maximum suppression detector
print("Running synthetic test: medium_108nm__nms_detector");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/medium_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Non-maximum suppression] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_108nm__nms_detector_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_108nm__nms_detector_imagej_timing.txt");
print(f, "medium_108nm__nms_detector," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test medium_108nm__nms_detector complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: medium_108nm__centroid_detector ===
// Dataset: Medium density, 108nm pixel  Algorithm: Centroid of connected components detector
print("Running synthetic test: medium_108nm__centroid_detector");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/medium_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Centroid of connected components] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_108nm__centroid_detector_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_108nm__centroid_detector_imagej_timing.txt");
print(f, "medium_108nm__centroid_detector," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test medium_108nm__centroid_detector complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: medium_108nm__lsq_fitting ===
// Dataset: Medium density, 108nm pixel  Algorithm: Least squares fitting
print("Running synthetic test: medium_108nm__lsq_fitting");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/medium_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_108nm__lsq_fitting_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_108nm__lsq_fitting_imagej_timing.txt");
print(f, "medium_108nm__lsq_fitting," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test medium_108nm__lsq_fitting complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: medium_108nm__mle_fitting ===
// Dataset: Medium density, 108nm pixel  Algorithm: Maximum likelihood estimation fitting
print("Running synthetic test: medium_108nm__mle_fitting");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/medium_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Maximum likelihood] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_108nm__mle_fitting_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_108nm__mle_fitting_imagej_timing.txt");
print(f, "medium_108nm__mle_fitting," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test medium_108nm__mle_fitting complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: medium_108nm__psf_gaussian ===
// Dataset: Medium density, 108nm pixel  Algorithm: PSF: Gaussian (non-integrated) with WLSQ
print("Running synthetic test: medium_108nm__psf_gaussian");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/medium_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_108nm__psf_gaussian_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_108nm__psf_gaussian_imagej_timing.txt");
print(f, "medium_108nm__psf_gaussian," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test medium_108nm__psf_gaussian complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: medium_108nm__radial_symmetry ===
// Dataset: Medium density, 108nm pixel  Algorithm: Radial symmetry estimator
print("Running synthetic test: medium_108nm__radial_symmetry");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/medium_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[Radial symmetry] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_108nm__radial_symmetry_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_108nm__radial_symmetry_imagej_timing.txt");
print(f, "medium_108nm__radial_symmetry," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test medium_108nm__radial_symmetry complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: medium_108nm__mfa_enabled ===
// Dataset: Medium density, 108nm pixel  Algorithm: Multi-emitter fitting analysis enabled
print("Running synthetic test: medium_108nm__mfa_enabled");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/medium_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=true keep_same_intensity=true nmax=5 fixed_intensity=false expected_intensity=500:2500 pvalue=1.0E-6 renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_108nm__mfa_enabled_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_108nm__mfa_enabled_imagej_timing.txt");
print(f, "medium_108nm__mfa_enabled," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test medium_108nm__mfa_enabled complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: medium_108nm__high_threshold ===
// Dataset: Medium density, 108nm pixel  Algorithm: Higher detection threshold (2x std)
print("Running synthetic test: medium_108nm__high_threshold");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/medium_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=2*std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_108nm__high_threshold_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_108nm__high_threshold_imagej_timing.txt");
print(f, "medium_108nm__high_threshold," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test medium_108nm__high_threshold complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: medium_108nm__fitradius_5 ===
// Dataset: Medium density, 108nm pixel  Algorithm: Larger fit radius (5 pixels)
print("Running synthetic test: medium_108nm__fitradius_5");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/medium_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=5 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_108nm__fitradius_5_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_108nm__fitradius_5_imagej_timing.txt");
print(f, "medium_108nm__fitradius_5," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test medium_108nm__fitradius_5 complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: dense_108nm__wavelet_default ===
// Dataset: Dense molecules, 108nm pixel  Algorithm: Wavelet B-Spline filter, default settings
print("Running synthetic test: dense_108nm__wavelet_default");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/dense_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=4-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/dense_108nm__wavelet_default_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/dense_108nm__wavelet_default_imagej_timing.txt");
print(f, "dense_108nm__wavelet_default," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test dense_108nm__wavelet_default complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: dense_108nm__wavelet_scale4_order5 ===
// Dataset: Dense molecules, 108nm pixel  Algorithm: Wavelet filter with scale=4, order=5
print("Running synthetic test: dense_108nm__wavelet_scale4_order5");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/dense_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=4.0 order=5 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/dense_108nm__wavelet_scale4_order5_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/dense_108nm__wavelet_scale4_order5_imagej_timing.txt");
print(f, "dense_108nm__wavelet_scale4_order5," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test dense_108nm__wavelet_scale4_order5 complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: dense_108nm__dog_filter ===
// Dataset: Dense molecules, 108nm pixel  Algorithm: Difference of Gaussians filter
print("Running synthetic test: dense_108nm__dog_filter");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/dense_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Difference-of-Gaussians filter] sigma1=1.0 sigma2=1.6 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/dense_108nm__dog_filter_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/dense_108nm__dog_filter_imagej_timing.txt");
print(f, "dense_108nm__dog_filter," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test dense_108nm__dog_filter complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: dense_108nm__gaussian_filter ===
// Dataset: Dense molecules, 108nm pixel  Algorithm: Gaussian (lowered) filter
print("Running synthetic test: dense_108nm__gaussian_filter");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/dense_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Lowered Gaussian filter] sigma=1.6 detector=[Local maximum] connectivity=8-neighbourhood threshold=1.5*std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/dense_108nm__gaussian_filter_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/dense_108nm__gaussian_filter_imagej_timing.txt");
print(f, "dense_108nm__gaussian_filter," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test dense_108nm__gaussian_filter complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: dense_108nm__nms_detector ===
// Dataset: Dense molecules, 108nm pixel  Algorithm: Non-maximum suppression detector
print("Running synthetic test: dense_108nm__nms_detector");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/dense_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Non-maximum suppression] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/dense_108nm__nms_detector_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/dense_108nm__nms_detector_imagej_timing.txt");
print(f, "dense_108nm__nms_detector," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test dense_108nm__nms_detector complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: dense_108nm__centroid_detector ===
// Dataset: Dense molecules, 108nm pixel  Algorithm: Centroid of connected components detector
print("Running synthetic test: dense_108nm__centroid_detector");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/dense_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Centroid of connected components] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/dense_108nm__centroid_detector_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/dense_108nm__centroid_detector_imagej_timing.txt");
print(f, "dense_108nm__centroid_detector," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test dense_108nm__centroid_detector complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: dense_108nm__lsq_fitting ===
// Dataset: Dense molecules, 108nm pixel  Algorithm: Least squares fitting
print("Running synthetic test: dense_108nm__lsq_fitting");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/dense_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/dense_108nm__lsq_fitting_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/dense_108nm__lsq_fitting_imagej_timing.txt");
print(f, "dense_108nm__lsq_fitting," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test dense_108nm__lsq_fitting complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: dense_108nm__mle_fitting ===
// Dataset: Dense molecules, 108nm pixel  Algorithm: Maximum likelihood estimation fitting
print("Running synthetic test: dense_108nm__mle_fitting");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/dense_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Maximum likelihood] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/dense_108nm__mle_fitting_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/dense_108nm__mle_fitting_imagej_timing.txt");
print(f, "dense_108nm__mle_fitting," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test dense_108nm__mle_fitting complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: dense_108nm__psf_gaussian ===
// Dataset: Dense molecules, 108nm pixel  Algorithm: PSF: Gaussian (non-integrated) with WLSQ
print("Running synthetic test: dense_108nm__psf_gaussian");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/dense_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/dense_108nm__psf_gaussian_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/dense_108nm__psf_gaussian_imagej_timing.txt");
print(f, "dense_108nm__psf_gaussian," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test dense_108nm__psf_gaussian complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: dense_108nm__radial_symmetry ===
// Dataset: Dense molecules, 108nm pixel  Algorithm: Radial symmetry estimator
print("Running synthetic test: dense_108nm__radial_symmetry");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/dense_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[Radial symmetry] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/dense_108nm__radial_symmetry_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/dense_108nm__radial_symmetry_imagej_timing.txt");
print(f, "dense_108nm__radial_symmetry," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test dense_108nm__radial_symmetry complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: dense_108nm__mfa_enabled ===
// Dataset: Dense molecules, 108nm pixel  Algorithm: Multi-emitter fitting analysis enabled
print("Running synthetic test: dense_108nm__mfa_enabled");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/dense_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=true keep_same_intensity=true nmax=5 fixed_intensity=false expected_intensity=500:2500 pvalue=1.0E-6 renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/dense_108nm__mfa_enabled_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/dense_108nm__mfa_enabled_imagej_timing.txt");
print(f, "dense_108nm__mfa_enabled," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test dense_108nm__mfa_enabled complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: dense_108nm__high_threshold ===
// Dataset: Dense molecules, 108nm pixel  Algorithm: Higher detection threshold (2x std)
print("Running synthetic test: dense_108nm__high_threshold");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/dense_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=2*std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/dense_108nm__high_threshold_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/dense_108nm__high_threshold_imagej_timing.txt");
print(f, "dense_108nm__high_threshold," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test dense_108nm__high_threshold complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: dense_108nm__fitradius_5 ===
// Dataset: Dense molecules, 108nm pixel  Algorithm: Larger fit radius (5 pixels)
print("Running synthetic test: dense_108nm__fitradius_5");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/dense_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=5 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/dense_108nm__fitradius_5_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/dense_108nm__fitradius_5_imagej_timing.txt");
print(f, "dense_108nm__fitradius_5," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test dense_108nm__fitradius_5 complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: low_snr_108nm__wavelet_default ===
// Dataset: Low SNR (few photons, high background)  Algorithm: Wavelet B-Spline filter, default settings
print("Running synthetic test: low_snr_108nm__wavelet_default");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/low_snr_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=4-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/low_snr_108nm__wavelet_default_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/low_snr_108nm__wavelet_default_imagej_timing.txt");
print(f, "low_snr_108nm__wavelet_default," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test low_snr_108nm__wavelet_default complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: low_snr_108nm__wavelet_scale4_order5 ===
// Dataset: Low SNR (few photons, high background)  Algorithm: Wavelet filter with scale=4, order=5
print("Running synthetic test: low_snr_108nm__wavelet_scale4_order5");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/low_snr_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=4.0 order=5 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/low_snr_108nm__wavelet_scale4_order5_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/low_snr_108nm__wavelet_scale4_order5_imagej_timing.txt");
print(f, "low_snr_108nm__wavelet_scale4_order5," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test low_snr_108nm__wavelet_scale4_order5 complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: low_snr_108nm__dog_filter ===
// Dataset: Low SNR (few photons, high background)  Algorithm: Difference of Gaussians filter
print("Running synthetic test: low_snr_108nm__dog_filter");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/low_snr_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Difference-of-Gaussians filter] sigma1=1.0 sigma2=1.6 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/low_snr_108nm__dog_filter_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/low_snr_108nm__dog_filter_imagej_timing.txt");
print(f, "low_snr_108nm__dog_filter," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test low_snr_108nm__dog_filter complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: low_snr_108nm__gaussian_filter ===
// Dataset: Low SNR (few photons, high background)  Algorithm: Gaussian (lowered) filter
print("Running synthetic test: low_snr_108nm__gaussian_filter");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/low_snr_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Lowered Gaussian filter] sigma=1.6 detector=[Local maximum] connectivity=8-neighbourhood threshold=1.5*std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/low_snr_108nm__gaussian_filter_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/low_snr_108nm__gaussian_filter_imagej_timing.txt");
print(f, "low_snr_108nm__gaussian_filter," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test low_snr_108nm__gaussian_filter complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: low_snr_108nm__nms_detector ===
// Dataset: Low SNR (few photons, high background)  Algorithm: Non-maximum suppression detector
print("Running synthetic test: low_snr_108nm__nms_detector");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/low_snr_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Non-maximum suppression] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/low_snr_108nm__nms_detector_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/low_snr_108nm__nms_detector_imagej_timing.txt");
print(f, "low_snr_108nm__nms_detector," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test low_snr_108nm__nms_detector complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: low_snr_108nm__centroid_detector ===
// Dataset: Low SNR (few photons, high background)  Algorithm: Centroid of connected components detector
print("Running synthetic test: low_snr_108nm__centroid_detector");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/low_snr_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Centroid of connected components] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/low_snr_108nm__centroid_detector_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/low_snr_108nm__centroid_detector_imagej_timing.txt");
print(f, "low_snr_108nm__centroid_detector," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test low_snr_108nm__centroid_detector complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: low_snr_108nm__lsq_fitting ===
// Dataset: Low SNR (few photons, high background)  Algorithm: Least squares fitting
print("Running synthetic test: low_snr_108nm__lsq_fitting");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/low_snr_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/low_snr_108nm__lsq_fitting_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/low_snr_108nm__lsq_fitting_imagej_timing.txt");
print(f, "low_snr_108nm__lsq_fitting," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test low_snr_108nm__lsq_fitting complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: low_snr_108nm__mle_fitting ===
// Dataset: Low SNR (few photons, high background)  Algorithm: Maximum likelihood estimation fitting
print("Running synthetic test: low_snr_108nm__mle_fitting");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/low_snr_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Maximum likelihood] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/low_snr_108nm__mle_fitting_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/low_snr_108nm__mle_fitting_imagej_timing.txt");
print(f, "low_snr_108nm__mle_fitting," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test low_snr_108nm__mle_fitting complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: low_snr_108nm__psf_gaussian ===
// Dataset: Low SNR (few photons, high background)  Algorithm: PSF: Gaussian (non-integrated) with WLSQ
print("Running synthetic test: low_snr_108nm__psf_gaussian");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/low_snr_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/low_snr_108nm__psf_gaussian_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/low_snr_108nm__psf_gaussian_imagej_timing.txt");
print(f, "low_snr_108nm__psf_gaussian," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test low_snr_108nm__psf_gaussian complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: low_snr_108nm__radial_symmetry ===
// Dataset: Low SNR (few photons, high background)  Algorithm: Radial symmetry estimator
print("Running synthetic test: low_snr_108nm__radial_symmetry");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/low_snr_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[Radial symmetry] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/low_snr_108nm__radial_symmetry_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/low_snr_108nm__radial_symmetry_imagej_timing.txt");
print(f, "low_snr_108nm__radial_symmetry," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test low_snr_108nm__radial_symmetry complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: low_snr_108nm__mfa_enabled ===
// Dataset: Low SNR (few photons, high background)  Algorithm: Multi-emitter fitting analysis enabled
print("Running synthetic test: low_snr_108nm__mfa_enabled");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/low_snr_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=true keep_same_intensity=true nmax=5 fixed_intensity=false expected_intensity=500:2500 pvalue=1.0E-6 renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/low_snr_108nm__mfa_enabled_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/low_snr_108nm__mfa_enabled_imagej_timing.txt");
print(f, "low_snr_108nm__mfa_enabled," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test low_snr_108nm__mfa_enabled complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: low_snr_108nm__high_threshold ===
// Dataset: Low SNR (few photons, high background)  Algorithm: Higher detection threshold (2x std)
print("Running synthetic test: low_snr_108nm__high_threshold");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/low_snr_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=2*std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/low_snr_108nm__high_threshold_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/low_snr_108nm__high_threshold_imagej_timing.txt");
print(f, "low_snr_108nm__high_threshold," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test low_snr_108nm__high_threshold complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: low_snr_108nm__fitradius_5 ===
// Dataset: Low SNR (few photons, high background)  Algorithm: Larger fit radius (5 pixels)
print("Running synthetic test: low_snr_108nm__fitradius_5");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/low_snr_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=5 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/low_snr_108nm__fitradius_5_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/low_snr_108nm__fitradius_5_imagej_timing.txt");
print(f, "low_snr_108nm__fitradius_5," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test low_snr_108nm__fitradius_5 complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: high_snr_108nm__wavelet_default ===
// Dataset: High SNR (many photons, low background)  Algorithm: Wavelet B-Spline filter, default settings
print("Running synthetic test: high_snr_108nm__wavelet_default");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/high_snr_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=4-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/high_snr_108nm__wavelet_default_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/high_snr_108nm__wavelet_default_imagej_timing.txt");
print(f, "high_snr_108nm__wavelet_default," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test high_snr_108nm__wavelet_default complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: high_snr_108nm__wavelet_scale4_order5 ===
// Dataset: High SNR (many photons, low background)  Algorithm: Wavelet filter with scale=4, order=5
print("Running synthetic test: high_snr_108nm__wavelet_scale4_order5");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/high_snr_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=4.0 order=5 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/high_snr_108nm__wavelet_scale4_order5_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/high_snr_108nm__wavelet_scale4_order5_imagej_timing.txt");
print(f, "high_snr_108nm__wavelet_scale4_order5," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test high_snr_108nm__wavelet_scale4_order5 complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: high_snr_108nm__dog_filter ===
// Dataset: High SNR (many photons, low background)  Algorithm: Difference of Gaussians filter
print("Running synthetic test: high_snr_108nm__dog_filter");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/high_snr_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Difference-of-Gaussians filter] sigma1=1.0 sigma2=1.6 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/high_snr_108nm__dog_filter_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/high_snr_108nm__dog_filter_imagej_timing.txt");
print(f, "high_snr_108nm__dog_filter," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test high_snr_108nm__dog_filter complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: high_snr_108nm__gaussian_filter ===
// Dataset: High SNR (many photons, low background)  Algorithm: Gaussian (lowered) filter
print("Running synthetic test: high_snr_108nm__gaussian_filter");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/high_snr_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Lowered Gaussian filter] sigma=1.6 detector=[Local maximum] connectivity=8-neighbourhood threshold=1.5*std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/high_snr_108nm__gaussian_filter_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/high_snr_108nm__gaussian_filter_imagej_timing.txt");
print(f, "high_snr_108nm__gaussian_filter," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test high_snr_108nm__gaussian_filter complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: high_snr_108nm__nms_detector ===
// Dataset: High SNR (many photons, low background)  Algorithm: Non-maximum suppression detector
print("Running synthetic test: high_snr_108nm__nms_detector");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/high_snr_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Non-maximum suppression] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/high_snr_108nm__nms_detector_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/high_snr_108nm__nms_detector_imagej_timing.txt");
print(f, "high_snr_108nm__nms_detector," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test high_snr_108nm__nms_detector complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: high_snr_108nm__centroid_detector ===
// Dataset: High SNR (many photons, low background)  Algorithm: Centroid of connected components detector
print("Running synthetic test: high_snr_108nm__centroid_detector");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/high_snr_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Centroid of connected components] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/high_snr_108nm__centroid_detector_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/high_snr_108nm__centroid_detector_imagej_timing.txt");
print(f, "high_snr_108nm__centroid_detector," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test high_snr_108nm__centroid_detector complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: high_snr_108nm__lsq_fitting ===
// Dataset: High SNR (many photons, low background)  Algorithm: Least squares fitting
print("Running synthetic test: high_snr_108nm__lsq_fitting");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/high_snr_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/high_snr_108nm__lsq_fitting_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/high_snr_108nm__lsq_fitting_imagej_timing.txt");
print(f, "high_snr_108nm__lsq_fitting," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test high_snr_108nm__lsq_fitting complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: high_snr_108nm__mle_fitting ===
// Dataset: High SNR (many photons, low background)  Algorithm: Maximum likelihood estimation fitting
print("Running synthetic test: high_snr_108nm__mle_fitting");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/high_snr_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Maximum likelihood] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/high_snr_108nm__mle_fitting_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/high_snr_108nm__mle_fitting_imagej_timing.txt");
print(f, "high_snr_108nm__mle_fitting," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test high_snr_108nm__mle_fitting complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: high_snr_108nm__psf_gaussian ===
// Dataset: High SNR (many photons, low background)  Algorithm: PSF: Gaussian (non-integrated) with WLSQ
print("Running synthetic test: high_snr_108nm__psf_gaussian");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/high_snr_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/high_snr_108nm__psf_gaussian_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/high_snr_108nm__psf_gaussian_imagej_timing.txt");
print(f, "high_snr_108nm__psf_gaussian," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test high_snr_108nm__psf_gaussian complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: high_snr_108nm__radial_symmetry ===
// Dataset: High SNR (many photons, low background)  Algorithm: Radial symmetry estimator
print("Running synthetic test: high_snr_108nm__radial_symmetry");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/high_snr_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[Radial symmetry] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/high_snr_108nm__radial_symmetry_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/high_snr_108nm__radial_symmetry_imagej_timing.txt");
print(f, "high_snr_108nm__radial_symmetry," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test high_snr_108nm__radial_symmetry complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: high_snr_108nm__mfa_enabled ===
// Dataset: High SNR (many photons, low background)  Algorithm: Multi-emitter fitting analysis enabled
print("Running synthetic test: high_snr_108nm__mfa_enabled");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/high_snr_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=true keep_same_intensity=true nmax=5 fixed_intensity=false expected_intensity=500:2500 pvalue=1.0E-6 renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/high_snr_108nm__mfa_enabled_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/high_snr_108nm__mfa_enabled_imagej_timing.txt");
print(f, "high_snr_108nm__mfa_enabled," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test high_snr_108nm__mfa_enabled complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: high_snr_108nm__high_threshold ===
// Dataset: High SNR (many photons, low background)  Algorithm: Higher detection threshold (2x std)
print("Running synthetic test: high_snr_108nm__high_threshold");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/high_snr_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=2*std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/high_snr_108nm__high_threshold_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/high_snr_108nm__high_threshold_imagej_timing.txt");
print(f, "high_snr_108nm__high_threshold," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test high_snr_108nm__high_threshold complete.");


run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Synthetic test: high_snr_108nm__fitradius_5 ===
// Dataset: High SNR (many photons, low background)  Algorithm: Larger fit radius (5 pixels)
print("Running synthetic test: high_snr_108nm__fitradius_5");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/high_snr_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=5 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/high_snr_108nm__fitradius_5_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/high_snr_108nm__fitradius_5_imagej_timing.txt");
print(f, "high_snr_108nm__fitradius_5," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test high_snr_108nm__fitradius_5 complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=267.0");

// === Synthetic test: sparse_60x__wavelet_default ===
// Dataset: Sparse molecules, 60x objective  Algorithm: Wavelet B-Spline filter, default settings
print("Running synthetic test: sparse_60x__wavelet_default");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_60x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=4-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.3 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_60x__wavelet_default_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_60x__wavelet_default_imagej_timing.txt");
print(f, "sparse_60x__wavelet_default," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_60x__wavelet_default complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=267.0");

// === Synthetic test: sparse_60x__wavelet_scale4_order5 ===
// Dataset: Sparse molecules, 60x objective  Algorithm: Wavelet filter with scale=4, order=5
print("Running synthetic test: sparse_60x__wavelet_scale4_order5");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_60x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=4.0 order=5 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.3 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_60x__wavelet_scale4_order5_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_60x__wavelet_scale4_order5_imagej_timing.txt");
print(f, "sparse_60x__wavelet_scale4_order5," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_60x__wavelet_scale4_order5 complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=267.0");

// === Synthetic test: sparse_60x__dog_filter ===
// Dataset: Sparse molecules, 60x objective  Algorithm: Difference of Gaussians filter
print("Running synthetic test: sparse_60x__dog_filter");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_60x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Difference-of-Gaussians filter] sigma1=1.0 sigma2=1.6 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.3 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_60x__dog_filter_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_60x__dog_filter_imagej_timing.txt");
print(f, "sparse_60x__dog_filter," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_60x__dog_filter complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=267.0");

// === Synthetic test: sparse_60x__gaussian_filter ===
// Dataset: Sparse molecules, 60x objective  Algorithm: Gaussian (lowered) filter
print("Running synthetic test: sparse_60x__gaussian_filter");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_60x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Lowered Gaussian filter] sigma=1.3 detector=[Local maximum] connectivity=8-neighbourhood threshold=1.5*std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.3 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_60x__gaussian_filter_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_60x__gaussian_filter_imagej_timing.txt");
print(f, "sparse_60x__gaussian_filter," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_60x__gaussian_filter complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=267.0");

// === Synthetic test: sparse_60x__nms_detector ===
// Dataset: Sparse molecules, 60x objective  Algorithm: Non-maximum suppression detector
print("Running synthetic test: sparse_60x__nms_detector");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_60x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Non-maximum suppression] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.3 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_60x__nms_detector_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_60x__nms_detector_imagej_timing.txt");
print(f, "sparse_60x__nms_detector," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_60x__nms_detector complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=267.0");

// === Synthetic test: sparse_60x__centroid_detector ===
// Dataset: Sparse molecules, 60x objective  Algorithm: Centroid of connected components detector
print("Running synthetic test: sparse_60x__centroid_detector");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_60x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Centroid of connected components] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.3 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_60x__centroid_detector_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_60x__centroid_detector_imagej_timing.txt");
print(f, "sparse_60x__centroid_detector," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_60x__centroid_detector complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=267.0");

// === Synthetic test: sparse_60x__lsq_fitting ===
// Dataset: Sparse molecules, 60x objective  Algorithm: Least squares fitting
print("Running synthetic test: sparse_60x__lsq_fitting");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_60x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.3 fitradius=3 method=[Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_60x__lsq_fitting_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_60x__lsq_fitting_imagej_timing.txt");
print(f, "sparse_60x__lsq_fitting," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_60x__lsq_fitting complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=267.0");

// === Synthetic test: sparse_60x__mle_fitting ===
// Dataset: Sparse molecules, 60x objective  Algorithm: Maximum likelihood estimation fitting
print("Running synthetic test: sparse_60x__mle_fitting");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_60x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.3 fitradius=3 method=[Maximum likelihood] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_60x__mle_fitting_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_60x__mle_fitting_imagej_timing.txt");
print(f, "sparse_60x__mle_fitting," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_60x__mle_fitting complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=267.0");

// === Synthetic test: sparse_60x__psf_gaussian ===
// Dataset: Sparse molecules, 60x objective  Algorithm: PSF: Gaussian (non-integrated) with WLSQ
print("Running synthetic test: sparse_60x__psf_gaussian");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_60x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Gaussian] sigma=1.3 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_60x__psf_gaussian_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_60x__psf_gaussian_imagej_timing.txt");
print(f, "sparse_60x__psf_gaussian," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_60x__psf_gaussian complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=267.0");

// === Synthetic test: sparse_60x__radial_symmetry ===
// Dataset: Sparse molecules, 60x objective  Algorithm: Radial symmetry estimator
print("Running synthetic test: sparse_60x__radial_symmetry");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_60x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[Radial symmetry] sigma=1.3 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_60x__radial_symmetry_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_60x__radial_symmetry_imagej_timing.txt");
print(f, "sparse_60x__radial_symmetry," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_60x__radial_symmetry complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=267.0");

// === Synthetic test: sparse_60x__mfa_enabled ===
// Dataset: Sparse molecules, 60x objective  Algorithm: Multi-emitter fitting analysis enabled
print("Running synthetic test: sparse_60x__mfa_enabled");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_60x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.3 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=true keep_same_intensity=true nmax=5 fixed_intensity=false expected_intensity=500:2500 pvalue=1.0E-6 renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_60x__mfa_enabled_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_60x__mfa_enabled_imagej_timing.txt");
print(f, "sparse_60x__mfa_enabled," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_60x__mfa_enabled complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=267.0");

// === Synthetic test: sparse_60x__high_threshold ===
// Dataset: Sparse molecules, 60x objective  Algorithm: Higher detection threshold (2x std)
print("Running synthetic test: sparse_60x__high_threshold");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_60x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=2*std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.3 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_60x__high_threshold_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_60x__high_threshold_imagej_timing.txt");
print(f, "sparse_60x__high_threshold," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_60x__high_threshold complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=267.0");

// === Synthetic test: sparse_60x__fitradius_5 ===
// Dataset: Sparse molecules, 60x objective  Algorithm: Larger fit radius (5 pixels)
print("Running synthetic test: sparse_60x__fitradius_5");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_60x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.3 fitradius=5 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_60x__fitradius_5_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_60x__fitradius_5_imagej_timing.txt");
print(f, "sparse_60x__fitradius_5," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_60x__fitradius_5 complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=160.0");

// === Synthetic test: sparse_100x__wavelet_default ===
// Dataset: Sparse molecules, 100x objective  Algorithm: Wavelet B-Spline filter, default settings
print("Running synthetic test: sparse_100x__wavelet_default");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_100x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=4-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.4 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_100x__wavelet_default_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_100x__wavelet_default_imagej_timing.txt");
print(f, "sparse_100x__wavelet_default," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_100x__wavelet_default complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=160.0");

// === Synthetic test: sparse_100x__wavelet_scale4_order5 ===
// Dataset: Sparse molecules, 100x objective  Algorithm: Wavelet filter with scale=4, order=5
print("Running synthetic test: sparse_100x__wavelet_scale4_order5");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_100x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=4.0 order=5 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.4 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_100x__wavelet_scale4_order5_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_100x__wavelet_scale4_order5_imagej_timing.txt");
print(f, "sparse_100x__wavelet_scale4_order5," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_100x__wavelet_scale4_order5 complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=160.0");

// === Synthetic test: sparse_100x__dog_filter ===
// Dataset: Sparse molecules, 100x objective  Algorithm: Difference of Gaussians filter
print("Running synthetic test: sparse_100x__dog_filter");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_100x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Difference-of-Gaussians filter] sigma1=1.0 sigma2=1.6 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.4 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_100x__dog_filter_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_100x__dog_filter_imagej_timing.txt");
print(f, "sparse_100x__dog_filter," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_100x__dog_filter complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=160.0");

// === Synthetic test: sparse_100x__gaussian_filter ===
// Dataset: Sparse molecules, 100x objective  Algorithm: Gaussian (lowered) filter
print("Running synthetic test: sparse_100x__gaussian_filter");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_100x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Lowered Gaussian filter] sigma=1.4 detector=[Local maximum] connectivity=8-neighbourhood threshold=1.5*std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.4 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_100x__gaussian_filter_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_100x__gaussian_filter_imagej_timing.txt");
print(f, "sparse_100x__gaussian_filter," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_100x__gaussian_filter complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=160.0");

// === Synthetic test: sparse_100x__nms_detector ===
// Dataset: Sparse molecules, 100x objective  Algorithm: Non-maximum suppression detector
print("Running synthetic test: sparse_100x__nms_detector");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_100x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Non-maximum suppression] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.4 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_100x__nms_detector_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_100x__nms_detector_imagej_timing.txt");
print(f, "sparse_100x__nms_detector," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_100x__nms_detector complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=160.0");

// === Synthetic test: sparse_100x__centroid_detector ===
// Dataset: Sparse molecules, 100x objective  Algorithm: Centroid of connected components detector
print("Running synthetic test: sparse_100x__centroid_detector");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_100x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Centroid of connected components] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.4 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_100x__centroid_detector_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_100x__centroid_detector_imagej_timing.txt");
print(f, "sparse_100x__centroid_detector," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_100x__centroid_detector complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=160.0");

// === Synthetic test: sparse_100x__lsq_fitting ===
// Dataset: Sparse molecules, 100x objective  Algorithm: Least squares fitting
print("Running synthetic test: sparse_100x__lsq_fitting");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_100x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.4 fitradius=3 method=[Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_100x__lsq_fitting_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_100x__lsq_fitting_imagej_timing.txt");
print(f, "sparse_100x__lsq_fitting," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_100x__lsq_fitting complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=160.0");

// === Synthetic test: sparse_100x__mle_fitting ===
// Dataset: Sparse molecules, 100x objective  Algorithm: Maximum likelihood estimation fitting
print("Running synthetic test: sparse_100x__mle_fitting");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_100x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.4 fitradius=3 method=[Maximum likelihood] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_100x__mle_fitting_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_100x__mle_fitting_imagej_timing.txt");
print(f, "sparse_100x__mle_fitting," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_100x__mle_fitting complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=160.0");

// === Synthetic test: sparse_100x__psf_gaussian ===
// Dataset: Sparse molecules, 100x objective  Algorithm: PSF: Gaussian (non-integrated) with WLSQ
print("Running synthetic test: sparse_100x__psf_gaussian");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_100x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Gaussian] sigma=1.4 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_100x__psf_gaussian_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_100x__psf_gaussian_imagej_timing.txt");
print(f, "sparse_100x__psf_gaussian," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_100x__psf_gaussian complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=160.0");

// === Synthetic test: sparse_100x__radial_symmetry ===
// Dataset: Sparse molecules, 100x objective  Algorithm: Radial symmetry estimator
print("Running synthetic test: sparse_100x__radial_symmetry");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_100x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[Radial symmetry] sigma=1.4 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_100x__radial_symmetry_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_100x__radial_symmetry_imagej_timing.txt");
print(f, "sparse_100x__radial_symmetry," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_100x__radial_symmetry complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=160.0");

// === Synthetic test: sparse_100x__mfa_enabled ===
// Dataset: Sparse molecules, 100x objective  Algorithm: Multi-emitter fitting analysis enabled
print("Running synthetic test: sparse_100x__mfa_enabled");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_100x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.4 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=true keep_same_intensity=true nmax=5 fixed_intensity=false expected_intensity=500:2500 pvalue=1.0E-6 renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_100x__mfa_enabled_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_100x__mfa_enabled_imagej_timing.txt");
print(f, "sparse_100x__mfa_enabled," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_100x__mfa_enabled complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=160.0");

// === Synthetic test: sparse_100x__high_threshold ===
// Dataset: Sparse molecules, 100x objective  Algorithm: Higher detection threshold (2x std)
print("Running synthetic test: sparse_100x__high_threshold");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_100x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=2*std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.4 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_100x__high_threshold_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_100x__high_threshold_imagej_timing.txt");
print(f, "sparse_100x__high_threshold," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_100x__high_threshold complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=160.0");

// === Synthetic test: sparse_100x__fitradius_5 ===
// Dataset: Sparse molecules, 100x objective  Algorithm: Larger fit radius (5 pixels)
print("Running synthetic test: sparse_100x__fitradius_5");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_100x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.4 fitradius=5 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_100x__fitradius_5_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_100x__fitradius_5_imagej_timing.txt");
print(f, "sparse_100x__fitradius_5," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_100x__fitradius_5 complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=107.0");

// === Synthetic test: sparse_150x__wavelet_default ===
// Dataset: Sparse molecules, 150x TIRF  Algorithm: Wavelet B-Spline filter, default settings
print("Running synthetic test: sparse_150x__wavelet_default");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_150x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=4-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.4 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_150x__wavelet_default_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_150x__wavelet_default_imagej_timing.txt");
print(f, "sparse_150x__wavelet_default," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_150x__wavelet_default complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=107.0");

// === Synthetic test: sparse_150x__wavelet_scale4_order5 ===
// Dataset: Sparse molecules, 150x TIRF  Algorithm: Wavelet filter with scale=4, order=5
print("Running synthetic test: sparse_150x__wavelet_scale4_order5");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_150x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=4.0 order=5 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.4 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_150x__wavelet_scale4_order5_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_150x__wavelet_scale4_order5_imagej_timing.txt");
print(f, "sparse_150x__wavelet_scale4_order5," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_150x__wavelet_scale4_order5 complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=107.0");

// === Synthetic test: sparse_150x__dog_filter ===
// Dataset: Sparse molecules, 150x TIRF  Algorithm: Difference of Gaussians filter
print("Running synthetic test: sparse_150x__dog_filter");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_150x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Difference-of-Gaussians filter] sigma1=1.0 sigma2=1.6 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.4 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_150x__dog_filter_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_150x__dog_filter_imagej_timing.txt");
print(f, "sparse_150x__dog_filter," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_150x__dog_filter complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=107.0");

// === Synthetic test: sparse_150x__gaussian_filter ===
// Dataset: Sparse molecules, 150x TIRF  Algorithm: Gaussian (lowered) filter
print("Running synthetic test: sparse_150x__gaussian_filter");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_150x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Lowered Gaussian filter] sigma=1.4 detector=[Local maximum] connectivity=8-neighbourhood threshold=1.5*std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.4 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_150x__gaussian_filter_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_150x__gaussian_filter_imagej_timing.txt");
print(f, "sparse_150x__gaussian_filter," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_150x__gaussian_filter complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=107.0");

// === Synthetic test: sparse_150x__nms_detector ===
// Dataset: Sparse molecules, 150x TIRF  Algorithm: Non-maximum suppression detector
print("Running synthetic test: sparse_150x__nms_detector");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_150x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Non-maximum suppression] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.4 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_150x__nms_detector_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_150x__nms_detector_imagej_timing.txt");
print(f, "sparse_150x__nms_detector," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_150x__nms_detector complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=107.0");

// === Synthetic test: sparse_150x__centroid_detector ===
// Dataset: Sparse molecules, 150x TIRF  Algorithm: Centroid of connected components detector
print("Running synthetic test: sparse_150x__centroid_detector");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_150x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Centroid of connected components] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.4 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_150x__centroid_detector_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_150x__centroid_detector_imagej_timing.txt");
print(f, "sparse_150x__centroid_detector," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_150x__centroid_detector complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=107.0");

// === Synthetic test: sparse_150x__lsq_fitting ===
// Dataset: Sparse molecules, 150x TIRF  Algorithm: Least squares fitting
print("Running synthetic test: sparse_150x__lsq_fitting");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_150x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.4 fitradius=3 method=[Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_150x__lsq_fitting_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_150x__lsq_fitting_imagej_timing.txt");
print(f, "sparse_150x__lsq_fitting," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_150x__lsq_fitting complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=107.0");

// === Synthetic test: sparse_150x__mle_fitting ===
// Dataset: Sparse molecules, 150x TIRF  Algorithm: Maximum likelihood estimation fitting
print("Running synthetic test: sparse_150x__mle_fitting");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_150x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.4 fitradius=3 method=[Maximum likelihood] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_150x__mle_fitting_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_150x__mle_fitting_imagej_timing.txt");
print(f, "sparse_150x__mle_fitting," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_150x__mle_fitting complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=107.0");

// === Synthetic test: sparse_150x__psf_gaussian ===
// Dataset: Sparse molecules, 150x TIRF  Algorithm: PSF: Gaussian (non-integrated) with WLSQ
print("Running synthetic test: sparse_150x__psf_gaussian");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_150x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Gaussian] sigma=1.4 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_150x__psf_gaussian_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_150x__psf_gaussian_imagej_timing.txt");
print(f, "sparse_150x__psf_gaussian," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_150x__psf_gaussian complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=107.0");

// === Synthetic test: sparse_150x__radial_symmetry ===
// Dataset: Sparse molecules, 150x TIRF  Algorithm: Radial symmetry estimator
print("Running synthetic test: sparse_150x__radial_symmetry");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_150x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[Radial symmetry] sigma=1.4 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_150x__radial_symmetry_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_150x__radial_symmetry_imagej_timing.txt");
print(f, "sparse_150x__radial_symmetry," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_150x__radial_symmetry complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=107.0");

// === Synthetic test: sparse_150x__mfa_enabled ===
// Dataset: Sparse molecules, 150x TIRF  Algorithm: Multi-emitter fitting analysis enabled
print("Running synthetic test: sparse_150x__mfa_enabled");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_150x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.4 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=true keep_same_intensity=true nmax=5 fixed_intensity=false expected_intensity=500:2500 pvalue=1.0E-6 renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_150x__mfa_enabled_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_150x__mfa_enabled_imagej_timing.txt");
print(f, "sparse_150x__mfa_enabled," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_150x__mfa_enabled complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=107.0");

// === Synthetic test: sparse_150x__high_threshold ===
// Dataset: Sparse molecules, 150x TIRF  Algorithm: Higher detection threshold (2x std)
print("Running synthetic test: sparse_150x__high_threshold");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_150x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=2*std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.4 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_150x__high_threshold_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_150x__high_threshold_imagej_timing.txt");
print(f, "sparse_150x__high_threshold," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_150x__high_threshold complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.9 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=107.0");

// === Synthetic test: sparse_150x__fitradius_5 ===
// Dataset: Sparse molecules, 150x TIRF  Algorithm: Larger fit radius (5 pixels)
print("Running synthetic test: sparse_150x__fitradius_5");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_150x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.4 fitradius=5 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_150x__fitradius_5_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_150x__fitradius_5_imagej_timing.txt");
print(f, "sparse_150x__fitradius_5," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_150x__fitradius_5 complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.8 isemgain=false photons2adu=0.5 gainem=1.0 pixelsize=160.0");

// === Synthetic test: medium_scmos_100x__wavelet_default ===
// Dataset: Medium density, sCMOS camera, 100x  Algorithm: Wavelet B-Spline filter, default settings
print("Running synthetic test: medium_scmos_100x__wavelet_default");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/medium_scmos_100x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=4-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.4 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_scmos_100x__wavelet_default_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_scmos_100x__wavelet_default_imagej_timing.txt");
print(f, "medium_scmos_100x__wavelet_default," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test medium_scmos_100x__wavelet_default complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.8 isemgain=false photons2adu=0.5 gainem=1.0 pixelsize=160.0");

// === Synthetic test: medium_scmos_100x__wavelet_scale4_order5 ===
// Dataset: Medium density, sCMOS camera, 100x  Algorithm: Wavelet filter with scale=4, order=5
print("Running synthetic test: medium_scmos_100x__wavelet_scale4_order5");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/medium_scmos_100x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=4.0 order=5 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.4 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_scmos_100x__wavelet_scale4_order5_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_scmos_100x__wavelet_scale4_order5_imagej_timing.txt");
print(f, "medium_scmos_100x__wavelet_scale4_order5," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test medium_scmos_100x__wavelet_scale4_order5 complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.8 isemgain=false photons2adu=0.5 gainem=1.0 pixelsize=160.0");

// === Synthetic test: medium_scmos_100x__dog_filter ===
// Dataset: Medium density, sCMOS camera, 100x  Algorithm: Difference of Gaussians filter
print("Running synthetic test: medium_scmos_100x__dog_filter");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/medium_scmos_100x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Difference-of-Gaussians filter] sigma1=1.0 sigma2=1.6 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.4 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_scmos_100x__dog_filter_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_scmos_100x__dog_filter_imagej_timing.txt");
print(f, "medium_scmos_100x__dog_filter," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test medium_scmos_100x__dog_filter complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.8 isemgain=false photons2adu=0.5 gainem=1.0 pixelsize=160.0");

// === Synthetic test: medium_scmos_100x__gaussian_filter ===
// Dataset: Medium density, sCMOS camera, 100x  Algorithm: Gaussian (lowered) filter
print("Running synthetic test: medium_scmos_100x__gaussian_filter");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/medium_scmos_100x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Lowered Gaussian filter] sigma=1.4 detector=[Local maximum] connectivity=8-neighbourhood threshold=1.5*std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.4 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_scmos_100x__gaussian_filter_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_scmos_100x__gaussian_filter_imagej_timing.txt");
print(f, "medium_scmos_100x__gaussian_filter," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test medium_scmos_100x__gaussian_filter complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.8 isemgain=false photons2adu=0.5 gainem=1.0 pixelsize=160.0");

// === Synthetic test: medium_scmos_100x__nms_detector ===
// Dataset: Medium density, sCMOS camera, 100x  Algorithm: Non-maximum suppression detector
print("Running synthetic test: medium_scmos_100x__nms_detector");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/medium_scmos_100x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Non-maximum suppression] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.4 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_scmos_100x__nms_detector_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_scmos_100x__nms_detector_imagej_timing.txt");
print(f, "medium_scmos_100x__nms_detector," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test medium_scmos_100x__nms_detector complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.8 isemgain=false photons2adu=0.5 gainem=1.0 pixelsize=160.0");

// === Synthetic test: medium_scmos_100x__centroid_detector ===
// Dataset: Medium density, sCMOS camera, 100x  Algorithm: Centroid of connected components detector
print("Running synthetic test: medium_scmos_100x__centroid_detector");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/medium_scmos_100x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Centroid of connected components] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.4 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_scmos_100x__centroid_detector_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_scmos_100x__centroid_detector_imagej_timing.txt");
print(f, "medium_scmos_100x__centroid_detector," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test medium_scmos_100x__centroid_detector complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.8 isemgain=false photons2adu=0.5 gainem=1.0 pixelsize=160.0");

// === Synthetic test: medium_scmos_100x__lsq_fitting ===
// Dataset: Medium density, sCMOS camera, 100x  Algorithm: Least squares fitting
print("Running synthetic test: medium_scmos_100x__lsq_fitting");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/medium_scmos_100x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.4 fitradius=3 method=[Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_scmos_100x__lsq_fitting_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_scmos_100x__lsq_fitting_imagej_timing.txt");
print(f, "medium_scmos_100x__lsq_fitting," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test medium_scmos_100x__lsq_fitting complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.8 isemgain=false photons2adu=0.5 gainem=1.0 pixelsize=160.0");

// === Synthetic test: medium_scmos_100x__mle_fitting ===
// Dataset: Medium density, sCMOS camera, 100x  Algorithm: Maximum likelihood estimation fitting
print("Running synthetic test: medium_scmos_100x__mle_fitting");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/medium_scmos_100x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.4 fitradius=3 method=[Maximum likelihood] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_scmos_100x__mle_fitting_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_scmos_100x__mle_fitting_imagej_timing.txt");
print(f, "medium_scmos_100x__mle_fitting," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test medium_scmos_100x__mle_fitting complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.8 isemgain=false photons2adu=0.5 gainem=1.0 pixelsize=160.0");

// === Synthetic test: medium_scmos_100x__psf_gaussian ===
// Dataset: Medium density, sCMOS camera, 100x  Algorithm: PSF: Gaussian (non-integrated) with WLSQ
print("Running synthetic test: medium_scmos_100x__psf_gaussian");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/medium_scmos_100x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Gaussian] sigma=1.4 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_scmos_100x__psf_gaussian_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_scmos_100x__psf_gaussian_imagej_timing.txt");
print(f, "medium_scmos_100x__psf_gaussian," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test medium_scmos_100x__psf_gaussian complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.8 isemgain=false photons2adu=0.5 gainem=1.0 pixelsize=160.0");

// === Synthetic test: medium_scmos_100x__radial_symmetry ===
// Dataset: Medium density, sCMOS camera, 100x  Algorithm: Radial symmetry estimator
print("Running synthetic test: medium_scmos_100x__radial_symmetry");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/medium_scmos_100x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[Radial symmetry] sigma=1.4 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_scmos_100x__radial_symmetry_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_scmos_100x__radial_symmetry_imagej_timing.txt");
print(f, "medium_scmos_100x__radial_symmetry," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test medium_scmos_100x__radial_symmetry complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.8 isemgain=false photons2adu=0.5 gainem=1.0 pixelsize=160.0");

// === Synthetic test: medium_scmos_100x__mfa_enabled ===
// Dataset: Medium density, sCMOS camera, 100x  Algorithm: Multi-emitter fitting analysis enabled
print("Running synthetic test: medium_scmos_100x__mfa_enabled");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/medium_scmos_100x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.4 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=true keep_same_intensity=true nmax=5 fixed_intensity=false expected_intensity=500:2500 pvalue=1.0E-6 renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_scmos_100x__mfa_enabled_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_scmos_100x__mfa_enabled_imagej_timing.txt");
print(f, "medium_scmos_100x__mfa_enabled," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test medium_scmos_100x__mfa_enabled complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.8 isemgain=false photons2adu=0.5 gainem=1.0 pixelsize=160.0");

// === Synthetic test: medium_scmos_100x__high_threshold ===
// Dataset: Medium density, sCMOS camera, 100x  Algorithm: Higher detection threshold (2x std)
print("Running synthetic test: medium_scmos_100x__high_threshold");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/medium_scmos_100x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=2*std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.4 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_scmos_100x__high_threshold_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_scmos_100x__high_threshold_imagej_timing.txt");
print(f, "medium_scmos_100x__high_threshold," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test medium_scmos_100x__high_threshold complete.");


run("Camera setup", "offset=100.0 quantumefficiency=0.8 isemgain=false photons2adu=0.5 gainem=1.0 pixelsize=160.0");

// === Synthetic test: medium_scmos_100x__fitradius_5 ===
// Dataset: Medium density, sCMOS camera, 100x  Algorithm: Larger fit radius (5 pixels)
print("Running synthetic test: medium_scmos_100x__fitradius_5");
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/medium_scmos_100x.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.4 fitradius=5 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_scmos_100x__fitradius_5_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/medium_scmos_100x__fitradius_5_imagej_timing.txt");
print(f, "medium_scmos_100x__fitradius_5," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test medium_scmos_100x__fitradius_5 complete.");
