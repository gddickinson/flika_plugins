run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

// === Test: mfa_enabled ===
// Multi-emitter fitting analysis enabled
print("Running test: mfa_enabled");
// Clear previous thunderSTORM results to prevent data carryover
if (isOpen("ThunderSTORM: results")) {
    run("Show results table", "action=reset");
}
wait(200);
run("Bio-Formats Importer", "open=[/Users/george/.FLIKA/plugins/spt_batch_analysis/test_data/real/Endothelial_NonBapta_bin10_crop.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=true keep_same_intensity=true nmax=5 fixed_intensity=false expected_intensity=500:2500 pvalue=1.0E-6 renderer=[No Renderer] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/.FLIKA/plugins/spt_batch_analysis/tests/comparison/results/imagej_results/mfa_enabled_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
// Save timing to file
f = File.open("/Users/george/.FLIKA/plugins/spt_batch_analysis/tests/comparison/results/imagej_results/mfa_enabled_imagej_timing.txt");
print(f, "mfa_enabled," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test mfa_enabled complete.");
