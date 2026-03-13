// === Synthetic test: sparse_108nm ===
// Sparse molecules, 108nm pixel (match real data)
run("Camera setup", "offset=100.0 quantumefficiency=1.0 isemgain=true photons2adu=3.6 gainem=100.0 pixelsize=108.0");

print("Running synthetic test: sparse_108nm");
if (isOpen("ThunderSTORM: results")) {
    selectWindow("ThunderSTORM: results");
    run("Close");
}
wait(500);
run("Bio-Formats Importer", "open=[/Users/george/claude_test/spt_batch_analysis/test_data/synthetic/sparse_108nm.tif] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.2 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[No Renderer]");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_108nm_imagej.csv] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("/Users/george/claude_test/spt_batch_analysis/tests/synthetic/results/imagej_results/sparse_108nm_imagej_timing.txt");
print(f, "sparse_108nm," + elapsed_ms);
File.close(f);
while (nImages>0) {
    selectImage(nImages);
    close();
}
print("Test sparse_108nm complete.");
