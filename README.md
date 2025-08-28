Laser Speckle Contrast Imaging Pipeline


NOTES/SETUP: 
• The code was tested in MatLab 2017b utilizes MatConvNet-1.0-Beta23 developed by the MatConvNet Team (see http://www.vlfeat.org/matconvnet/). 
• Questions may be directed to Dr. Daniel Lipinski (dlipinski@mcw.edu, Principal Investigator) or Alex Tate (ajtate@mcw.edu, code developer).
• Several variables can exceed the default file size limit for saving the workplace; this limit may be saved or lines saving the workplace may be silenced.
• The code requires the "Computer Vision Toolbox" and "Signal Processing Toolbox" of MatLab. Our installed package list is appendixed under "MatLab Packages" for reference. 
• Before running, images should be named systematically (i.e. ImageName1.tif, ImageName2.tif,... ImagenameN.tif) within the same folder. The image calling script may need adjusted depending on image naming convention.
• The code has most extensively been tested with .tif images, and to a lesser extent .bmp images. This pipeline may be compatible with other images, however image compression may signficantly distort hemodynamic output. 
• IMPORTANT: Currently this pipeline expects artery-vein delineation to be performed and as such requires a minimum of two ROIs to be assigned with one being attributed as artery and another as vein. Future versions of the code will decouple vessel type delineation from hemodynamic analysis. If only global retinal flow hemoydynamics are needed, arbitrary ROIs may be assigned and attributed as artery and vein.


SECTION 1: Set Up and Image Uploading (<10000)
• Default image parameters for image calling are 1000 sequentially numbered .tif images with padded numbering; these parameters are all adjustable.
	• Pipeline has been validated to work with grayscale .bmp and .tif/.tiff files; other image file formats convertible to grayscale are expected to be compatible but need validated and may require conversion.
• Input whether data saving is desired. This choice toggles saving for the entire pipeline. Individual workplace or figure saving can be silenced using "%" before the corresponding lines.
• Input whether image cropping is desired. If so, draw a rectangle the loaded image, right click, and copy&paste coordinates into the command window; otherwise, indicate enter "0" in the command window
• QUALITY CHECK: confirm that pixel intensity histogram has an indicated proportion of >90% within the 5-55% range, with a right skew preferrable to a left skew (this pixel intensity range corresponds to the linear range of speckle contrast)
	• An increase in pixels <5% brightness is typically resolved by cropping the image to exclude peripheral regions of low illumination.
	• If illumination is uniform not sufficiently within linear range, adjustments are needed to laser power, exposure, and/or gain during image acquisition.	
• Output: histogram of pixel intensity with proportion within linear range


SECTION 2: Speckle and Flow Calculations with Spatial Averaging
• Continuous with Section 1, no input needed
• Adjustable parameters: 
	• ToggleMP4: changing the value to 0/1 turns off/on generation of a flow video file. Turned off by default because computationally slow.
• IMPORTANT NOTE: spatial neighborhood size has a non-linear, inversely proportional effect on speckle contrast and alters all downstream calculations. Default spatial window size is 5x5. It is suggested to avoid altering spatial window size without good reason and to only compare data sets analyzed with the same window size. 
• Output: flow video


SECTION 3: Sub-ROI Selection
• Input largest intra-breath range of frames (total must be even for downstream fourier analysis, so start and end frames should be odd-even or even-odd, not odd-odd or even-even)
• Input "1" if a sub-ROI (i.e. a blood vessel) needs defined, followed by outlining the ROI on the most recently opened flow image and double-clicking the image to confirm sub-ROI selection; repeat as necessary, inputting "0" when no additional ROIs are needed.
	• NOTE: currently the pipeline requires at least two sub-ROIs be defined here, with the expectation artery(s) and vein(s) are being defined for downstream delineation. Future versions of the pipeline plan to implement conditional/toggleable vessel-type delineation and decouple ROI selection from vessel delineation allowing for only global flow analysis or adaptation for other regional analyses (i.e. central vs peripheral, lesioned vs non-lesioned...)
• Output: temporally averaged flow image, sub-ROI selection


SECTION 4: Frequency Analysis, Flow Profile Delineation, and Signal Filtering
• Input the BIN NUMBER of the cardiac frequency (expected murine range: ~6-9Hz) using the power spectral density plot for reference; repeat for the 1st-3rd harmonics
• Output: T-plot of automatically assigned vessel type
• MANUAL CORROBORATION: vessel delineation (variable "FinalVesselType") should be manually corroborated by the reviewer, using -1 to denote veins and 1 to denote arteries
	• Vessel attribution from the T-plots should be corroborated with expectations from parallel arteriovenous pairing, relative vessel diameter, and relative flow dynamics (i.e. veins exhibit greater basal flow and lower pulsatility, arteries exhibit lower basal flow but greater pulsatility; arterial flow precedes venous flow)
	• External corroboration of vessel type may be desired in image sets with less certainty
	• The workplace is checkpointed here in the case vessel type attribution redone


SECTION 5: Pulse Recognition, Registration, and Averaging
• Pulses are identified within the arterial flow profile and then fed-forward to inform pulse identification in other flow profiles
• Looped inputs will be requested to add or remove pulse starts. Do so by inputting the frame number for the pulse start to be added/removed or by inputting 0 if no addition/removal is needed.
	• Example: the algorithm may occasionally incorrectly attribute the pulse starts to intra-pulse notches which are also periodically synchronous
• Looped inputs will be requested to remove noisy or asynchronous pulses, such as pulses overlapping with respiration artifact
	• Note: there are two passes of this algorithm - the first should be used to exclude pulses which overlap with movement artifact or other significant noise; the second displays temporally normalized pulses and should be used to exclude aysnchronous pulses
	• Note: regardless of if pulses are manually excluded, after this step any statistical outliers in amplitude or periodicity will be automatically excluded if not already excluded manually.


%% SECTION 6: Hemodynamic Analysis
• Following pulse identification and exclusion, pulses are averaged together for feature recognition (peaks and notches) and hemodynamic analysis
• There is opportunity to manually attribute a non-discrete shoulder peak using first and second derivative corroboration. Input first the flow profile number followed by the frame number of the shoulder peak.
	• Only fully discrete notches will be listed. In the case of missing notches, values are replicated and will removed from the exported hemodynamic summary
• Summary plots will be generated and saved for the included pulses, averaged flow profiles, and summed arterial and venous power spectrum density plots
• Hemodynamics will be summarized in a variable called "Summary" which can be copy and pasted into the provided excel sheet template "LSCI Hemodynamic Summary Template"
