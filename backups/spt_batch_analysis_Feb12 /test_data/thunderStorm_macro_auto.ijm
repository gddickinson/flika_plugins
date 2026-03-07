datapaths = newArray('/Users/george/Data/minimal_model/crop/Endothelial_NonBapta_bin10_crop.tif');
respaths = newArray('/Users/george/Data/minimal_model/crop/Endothelial_NonBapta_bin10_crop_locs.csv');
for (i=0; i  < datapaths.length; i++) {
 	//open(datapaths[i]);
 	run("Bio-Formats Importer", "open=" + datapaths[i] +" color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
 	run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=4-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=false renderer=[Averaged shifted histograms] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
 	run("Export results", "filepath=["+respaths[i]+"] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true id=true uncertainty=true frame=true");
 	//close();
    while (nImages>0) {
        selectImage(nImages);
        close();
    }
}