clear all; close all; clc;
%% SECTION 1: Set Up and Image Uploading (<10000)
% Step 1: Set Up
Save = input('Save all images and data? Yes(1) or No(0):   ');
    if Save == 1
        Expt = input('Experiment Title/Number (i.e. Expt##): ', 's'); 
        Group = input('Experimental Group: ', 's');       
        ID = input('Animal ID (i.e. M##/F##): ', 's');
        DataName = strcat(Expt, " ", Group, " ", ID);
    end

    if input('Is data set 1000 padded .tif images collected at 376.2Hz? Yes(1) or No(0):   ') == 1;
        NumImages = 1000; 
        FPS = 376.2; 
        ImageType = '.tif'; 
        Padding = 1; 
    else
        NumImages = input('How many images?: '); 
        FPS = input('What was the framerate?: ');
        ImageType = input('What is the image type? e.g. if file saved as Image###.tif, enter .tif:    ', 's');
        Padding = input('Do filenames need padded (i.e. first image = Image000)? Yes(1) or No(0):   ');
    end
    
    if Padding == 1; PaddingOrder = numel(num2str(NumImages-1)); 
    else; PaddingOrder = 0; end;
    
Stem = input('What is the image filename stem? e.g. if file saved as Test.tif, enter Test:    ', 's'); 
FirstImage = imread(sprintf('%s%0*d%s', Stem, PaddingOrder, 0, ImageType)); 
[Rows, Cols] = size(FirstImage);
   
% Step 2: Conditional Cropping
clc; imshow(mat2gray(FirstImage));    
    if input('Does image need cropped? Yes(1) or No(0):   ') == 1
        disp('Drag&draw a rectangle, right click, and copy coordinates.');
        figure(1); h = imrect; CropXY = input('Paste coordinates here: ');
        CropXY = [round(CropXY(1)), round(CropXY(2)), round(CropXY(3)), round(CropXY(4))];
        CropRows = [CropXY(2), CropXY(2) + CropXY(4)];
        CropColumns = [CropXY(1), CropXY(1) + CropXY(3)];
    else; CropRows = [1, Rows]; CropColumns = [1, Cols]; end;
CropHeight = CropRows(2) - CropRows(1) + 1;CropWidth = CropColumns(2) - CropColumns(1) + 1;
ImageStack = zeros(CropHeight, CropWidth, NumImages, 'like', FirstImage);

% Step 3: Image Calling
    for m = 0:NumImages-1 
        if Padding == 1; FileName = sprintf('%s%0*d%s', Stem, PaddingOrder, m, ImageType);
        else; FileName = sprintf('%s%d%s', Stem, m, ImageType); end; 
        Image = imread(FileName);
        CroppedImage = Image(CropRows(1):CropRows(2), CropColumns(1):CropColumns(2));
        ImageStack(:,:,m+1) = CroppedImage;
    end

clearvars -except DataName FPS ImageStack NumImages Save;

% Step 4: Linear Range and Bit Upshift
MaxIntensity = max(ImageStack(:));
    if MaxIntensity <= 255 
        ImageStack = bitshift(uint16(ImageStack),8); 
    elseif MaxIntensity <= 4096
        ImageStack = bitshift(ImageStack,4);
    end
InLinearRange = sum(3855 <= ImageStack(:) & ImageStack(:) <= 34695)/numel(ImageStack)*100;
BelowLinearRange = sum(ImageStack(:)<3855)/numel(ImageStack)*100;
AboveLinearRange = sum(ImageStack(:)>34695)/numel(ImageStack)*100;

figure
    hold on
        histogram(ImageStack);
        xlim([0,65536])
        line([3855 3855], ylim, 'Color', 'b', 'LineWidth', 2);
        line([34695 34695], ylim, 'Color', 'r', 'LineWidth', 2);
        annotation('textbox', [0.3, 0.833, 0.1, 0.1], 'String', sprintf('%.2f%%', InLinearRange), 'EdgeColor', 'none', 'FontSize', 12, 'FontWeight', 'bold');
        if exist('AboveLinearRange','var')
            annotation('textbox', [0.66, 0.833, 0.1, 0.1], 'String', sprintf('%.2f%%', AboveLinearRange), 'EdgeColor', 'none', 'FontSize', 12, 'FontWeight', 'bold','Color','r');
        end
    hold off

    if Save == 1
        saveas(gcf,strcat(DataName," Intensity Histogram.tif"));
    end   
    
clear MaxIntensity; clc; 

% SECTION 2: Speckle and Flow Calculations with Spatial Averaging
% Step 1: Set Up
IntensityMean = zeros(size(ImageStack)); 
IntensitySD = IntensityMean; Speckle = IntensityMean; Flow = IntensityMean;
Window = 5; Weight = ones(Window,Window)/(Window^2);

% Step 2: Spatial Processing
    for i = 1:NumImages
        frame = ImageStack(:,:,i);
        IntensityMean(:,:,i) = imfilter(frame, Weight,'replicate');
        IntensitySD(:,:,i) = stdfilt(frame, true(Window));
    end

% Step 3: Compute Speckle and Flow
Speckle = IntensitySD./IntensityMean; 
    Speckle(Speckle<0.001) = NaN; %avoids near inf's
Flow = 1./(Speckle.^2); 
    Flow(isinf(Flow)) = NaN; %removes inf's
TimeFlow = (0:NumImages-1)/FPS;
AvgFlow = sum(sum(Flow, 'omitnan'))./sum(sum(Flow~=0,'omitnan'));
BFI = reshape(AvgFlow, [1, NumImages]);

% Step 4: Write Output Video
ToggleMP4 = 0; %Toggle 0/1 to turn section off/on
    if ToggleMP4 == 1 && Save == 1;
        OutputVideo = VideoWriter(char(strcat(DataName, ' Flow.mp4')), 'MPEG-4');
        OutputVideo.FrameRate = 60; MovingFilter = 5; Buffer = floor(0.5*MovingFilter);
        open(OutputVideo);
        for i = (1+Buffer):(NumImages-Buffer)
                SumFrame = sum(double(Flow(:,:,i-Buffer:i+Buffer)), 3);
                AverageFrame = SumFrame / MovingFilter; AverageFrame = mat2gray(AverageFrame);
                writeVideo(OutputVideo, AverageFrame);
        end; close(OutputVideo);
    end
clear OutputVideo MovingFilter Buffer Sum Frame AverageFrame AvgFlow frame i AboveLinearRange BelowLinearRange IntensityMean IntensitySD Window Weight ToggleMP4
sound(sin(1:3000)); clc; 

%% SECTION 3: ROI Selection
% Step 1: Define Temporal Range
figure; plot(BFI);
TemporalRange = [input('Intra-breath start frame (odd): '), input('Intra-breath end frame (even): ')];
close(gcf);

% Step 2: Compute/plot temporal average and background
TempAvgK = mean(Speckle(:,:,TemporalRange(1):TemporalRange(2)), 3);
TempAvgFlow = mean(Flow(:,:,TemporalRange(1):TemporalRange(2)), 3);

OffVessel = TempAvgFlow < min(BFI(1, :));
    Window = 11; Weight = ones(Window, Window) / (Window^2);
    OffVessel = imfilter(OffVessel, Weight);
    StoreROI{1} = OffVessel;
% OnVessel = -(OffVessel-1);
    
figure; imshow(OffVessel);
if Save
    saveas(gcf, strcat(DataName, " Major Vessel Mask.tif"));
end   

% Step 3: Vessel ROI Selection 
% IN PROGRESS (DECOUPLE ROI FROM AV)
figure; imshow(mat2gray(TempAvgFlow));
ToggleROI = 1;

    if ToggleROI == 1
        Hold = 1;
        OverlayROI = repmat(mat2gray(TempAvgFlow), [1, 1, 3]);
        numROIs = 1; 
        ToggleAV = input('Delineate Vessel Type? Yes(1) or No(0): ');
    end

    while Hold == 1
        ROI = roipoly; StoreROI{numROIs+1} = ROI;
            ROI_boundaries = bwperim(ROI);
            OverlayROI(:,:,1) = OverlayROI(:,:,1) + ROI_boundaries;
            Stats = regionprops(ROI, 'Centroid'); Centroid = Stats.Centroid;
            OverlayROI = insertText(OverlayROI, Centroid, num2str(numROIs), 'TextColor', 'white', 'BoxOpacity', 0, 'FontSize', 36);
        figure(gcf); imshow(mat2gray(OverlayROI)); 
            Hold = input('Add Another Vessel ROI? Yes(1) or No(0): '); 
            if Hold ~= 1; break; end;
        numROIs = numROIs+1; clc;
    end

    for i = 1:size(StoreROI,2)
        ROI = StoreROI{i}; 
        SubsetFlow = ROI .* Flow;
        SubsetFlow(SubsetFlow==0) = NaN;
        SubsetFlow = sum(sum(SubsetFlow, 'omitnan'), 'omitnan') / sum(sum(ROI ~= 0, 'omitnan'),'omitnan');
        BFI(i+1, :) = reshape(SubsetFlow, [1, NumImages]);    
    end
    
    if Save == 1
        saveas(gcf,strcat(DataName," Temporal Average with ROIs.tif"));
    end     
    
clear ImageStack Speckle OffVessel Window Weight numROIs TempAvgK; clc;    
clear i Hold ROI ROI_boundaries Stats Centroid SubsetFlow TemporalRange; sound(sin(1:3000)); clc;

%% SECTION 4: Frequency Analysis - Vessel Delineation and Filtration
% Step 1: Calculate Pixel PSD's
    for i = 1:size(BFI,1)
        if i == 1
            N = NumImages;
            FFT = fft(Flow,[],3); 
            PSD = (abs(FFT).^2)/(FPS*N); %Crosscheck T = FPS*NumImages...
            PSD(:,:,2:end-1) = 2*PSD(:,:,2:end-1);
            PSD(:,:,1) = 0; PSD = PSD(:,:,1:(N/2)+1);
            FreqPSD = (0:FPS/N:FPS/2);
            LinearPSD(i,:) = squeeze(mean(mean(PSD,'omitnan'),'omitnan'))';
        else
            subsetPSD = StoreROI{i-1}.*PSD;
            subsetPSD(subsetPSD==0)=NaN;
            LinearPSD(i,:) = squeeze(mean(mean(subsetPSD,'omitnan'),'omitnan'))';
        end
    end
sound(sin(1:3000));

% Step 2: Identify Harmonics
figure
    subplot(2,1,1)
    hold on
        for i = 1:size(BFI,1)
            plot(LinearPSD(i,:))
        end
        xlabel('Bin Number')
    subplot(2,1,2)
    hold on
        for i = 1:size(BFI,1)
            plot(FreqPSD,LinearPSD(i,:))
        end
        xlabel ('Frequency (Hz)')
    hold off
        
NumHarmonics = 3; %input('How many distinct harmonics for global?: ')
    for i = 1:NumHarmonics+1
        figure(gcf)
        if i == 1; Harmonics(1,i) = input(strcat('Bin Number (X) of Cardiac Frequency: '));
        else Harmonics(1,i) = input(strcat('Bin Number (X) of Harmonic ',num2str(i-1), ': ')); end
        Harmonics(2,i) = (Harmonics(1,i)-1)*FPS/N;
        Harmonics(3,i) = LinearPSD(Harmonics(1,i));
    end
close(gcf) 

% Step 3: T Plots for Vessel Delineation
PSDmidpoint = ceil(size(PSD,3)/2); PSDnoise = mean(PSD(:,:,PSDmidpoint:end),3);
figure
    hold on
    for j = 1:3 
    % Calculates H, R, and T
        PSDframe = PSD(:,:,Harmonics(1,j)); %PSD(:,:,Harmonics(1,N)) to select nth Harmonic
            H_factor = PSDframe - PSDnoise;
            R_factor = PSDframe ./ PSDnoise;
        H_BG = StoreROI{1}.*H_factor; H_BG(H_BG==0) = NaN;
            Hmean_BG = mean(H_BG(:),'omitnan'); Hstd_BG = std(H_BG(:),'omitnan');
            vein = (H_factor < (Hmean_BG - Hstd_BG));
        R_BG = StoreROI{1}.*R_factor; R_BG(R_BG==0) = NaN;
            Rmean_BG = mean(R_BG(:),'omitnan'); Rstd_BG = std(R_BG(:),'omitnan');
            artery = (R_factor > (Rmean_BG + Rstd_BG));
        Ra = prctile(R_factor(StoreROI{1}~=1),90); Rv = prctile(R_factor(StoreROI{1}~=1),10);
            delineate = (2*R_factor-Ra-Rv)/(Ra+Rv);
            delineate(delineate>0) = delineate(delineate>0)/max(delineate(:)); %sets positive range to [0,1]
        combined = zeros(size(PSDframe));
            combined(vein==1 & artery~=1) = -1;
            combined(vein~=1 & artery==1) = 1;
            combined(combined==0)=delineate(combined==0);
        TPlot{j} = combined;
    % Spatially Average and Plot T
        combined = imfilter(combined,(ones(3,3)/9)); 
        subplot(2,3,j) %(2,length(Harmonics),j)
            imagesc(combined);
            axis image; axis off;
            cmap = [0 0 1; 0 0 0; 1 1 0];
            customMap = interp1([-1,0,1],cmap,linspace(-1,1,256),'pchip');
            colormap(customMap); 
    % Estimate and Plot Vessel (ROI) Classification
        VesselType = zeros(2,size(StoreROI,2));
        for i = 1:size(StoreROI,2)
            subTPlot = combined.*StoreROI{i};
            subTPlot(subTPlot==0)=NaN;
            VesselType(1,i) = nanmean(squeeze(subTPlot(:)));
                if VesselType(1,i) < VesselType(1,1)
                    VesselType(2,i) = -1;
                elseif VesselType(1,i) > VesselType(1,1)
                    VesselType(2,i) = 1;
                else VesselType(2,i) = 0;
                end
            combined(StoreROI{i}==1) = VesselType(2,i);
        end
        StoreVesselType(j,:) = VesselType(2,:);
        subplot(2,3,j+3) 
            imagesc(combined);
            axis image; axis off;
            cmap = [0 0 1; 0 0 0; 1 0 0];
            customMap = interp1([-1,0,1],cmap,linspace(-1,1,256),'pchip');
            colormap(customMap);              
    clear PSDframe H_factor R_factor H_BG Hmean_BG Hstd_BG R_BG Rmean_BG Rstd_BG artery vein;
    clear Ra Rv delineate combined ROIcombined ROImeanTPlot VesselType cmap customMap VesselType subTPlot;
    end
    hold off
    
    if Save == 1
        saveas(gcf, strcat(DataName, " Vessel Delineation.tif"));
    end     

% Step 4: Manual Vessel Corroboration
StoreVesselType
FinalVesselType = mode(StoreVesselType,1)
display('Manually Corroborate Vessel Types (Should Alternate)');

%% CHECKPOINT
clear N NumHarmonics i j l1 PSD PSDnoise; sound(sin(1:3000)); clc;
clear FFT subsetPSD Flow;
if Save == 1
    save(strcat(DataName,".mat")); display('Workplace has been saved as a checkpoint');
end     

%% Filtration and Flow Profile Delineation
%Step 5: Apply Buttersworth LowPass Filter
    %Note: For noisier data, increase Rp (range: 0.25-5) & Rs (range: 5-10) 
W = Harmonics(2,4)/(FPS/2); Rp = 5; Rs = 10;
    [n, Wn] = buttord(W, W*0.9, Rp, Rs); % Adjust 0.9 stopband edge if needed
    [b, a] = butter(n, Wn);
    for i = 1:size(BFI,1)
        FilteredBFI(i,:) = filtfilt(b, a, BFI(i,:));
    end
    
figure;
    hold on;
    for i = 1:size(BFI,1)
        plot(TimeFlow, BFI(i,:), 'k:', 'LineWidth', 1);
        plot(TimeFlow, FilteredBFI(i,:), 'LineWidth', 1.5);
    end
    l1 = legend;
    set(l1, 'FontSize', 10, 'Location', 'NorthEast');
    hold off;

clear W Rp Rs n Wn b a i l1;clc;

% Generating Flow Profiles
    % Note: All downstream figures, Global=Green, Background=Black, Arteries=Red, Veins=Blue
colors = {'g', 'k', 'r', 'b'}; 
titles = {'Global Flow', 'Background Flow', 'Arterial Flow', 'Venous Flow'}; 
ArterialFlow = []; VenousFlow = []; ArterialPSD = []; VenousPSD = [];
FlowProfiles = zeros(4,length(FilteredBFI)); 

figure;
    subplot(3,1,1) %Arterial Flow
        hold on
        for i = 1:length(FinalVesselType)
            if FinalVesselType(i) == 1;
                plot(TimeFlow(1,:), FilteredBFI(i+1,:),'k:','LineWidth', 1);
                ArterialFlow = [ArterialFlow;FilteredBFI(i+1,:)];
                ArterialPSD = [ArterialPSD;LinearPSD(i+1,:)];
            end
        end
            plot(TimeFlow(1,:),mean(ArterialFlow),'r','LineWidth',2);
            title('Arteries'); xlabel('Time (ms)'); ylabel('Flow, 1/K^2 (a.u.)');
        hold off
    subplot(3,1,3) %Venous Flow
        hold on
        for i = 1:length(FinalVesselType)
            if FinalVesselType(i) == -1;
                plot(TimeFlow(1,:), FilteredBFI(i+1,:),'k:','LineWidth', 1);
                VenousFlow = [VenousFlow;FilteredBFI(i+1,:)];
                VenousPSD = [VenousPSD;LinearPSD(i+1,:)];
            end
        end
            plot(TimeFlow(1,:),mean(VenousFlow),'b','LineWidth',2);
            title('Veins'); xlabel('Time (ms)'); ylabel('Flow, 1/K^2 (a.u.)');
        hold off
    subplot(3,1,2) %Flow Profiles
        hold on
            plot(TimeFlow(1,:), FilteredBFI(1,:),'g', 'LineWidth', 2);
            plot(TimeFlow(1,:), FilteredBFI(2,:),'k', 'LineWidth', 2);
            plot(TimeFlow(1,:),mean(ArterialFlow),'r','LineWidth',2);
            plot(TimeFlow(1,:),mean(VenousFlow),'b','LineWidth',2);
            title('Flow Profiles'); xlabel('Time (ms)'); ylabel('Flow, 1/K^2 (a.u.)');
        hold off
      
FlowProfiles(1,:) = FilteredBFI(1,:); %Calls Global Flow
FlowProfiles(2,:) = FilteredBFI(2,:); %Calls Background Flow
if exist('ArterialFlow','var'); FlowProfiles(3,:) = mean(ArterialFlow,1); end; %Arterial Flow Profile
if exist('VenousFlow','var');FlowProfiles(4,:) = mean(VenousFlow,1); end; %Venous Flow Profile
   
if Save == 1
    saveas(gcf, strcat(DataName, " Flow Profiles.tif"));
    saveas(gcf, strcat(DataName, " Flow Profiles.fig"));
end     
% clear i VenousFlow ArterialFlow;

%% CHECKPOINT
if Save == 1
    save(strcat(DataName,".mat")); display('Workplace has been saved as a checkpoint');
end     

%% SECTION 5: Calculating Pulse Starts (with Manual Corroboration)
% Find Preliminary PulseStarts in Arterial Flow Profile 
thresholdPI = floor(0.9 * (FPS / Harmonics(2,1))); % Adjust 90% as needed
[pksA,locsA] = findpeaks(-FlowProfiles(3,:), 'MinPeakDistance', thresholdPI, 'MinPeakProminence', 0.1);

    % Plots Arterial Flow Profile and PulseStart Candidates
    figure 
        hold on
            hWaveform = plot(FlowProfiles(3,:)); 
            hPulseStarts = plot(locsA,FlowProfiles(3,locsA), 'ko'); 
        hold off
    Insert = [FlowProfiles(3,locsA); locsA]; % Insert for PulseStart Corrections

    % Addition loop 
    add = 1;
    while add ~= 0
        figure(gcf); add = input('Add a PulseStart? No(0), or Yes (Input frame number): ');
        if add ~= 0
            addX = add; addY = FlowProfiles(3,addX); 
            Insert = [Insert, [addY; addX]]; 
            set(hPulseStarts, 'XData', Insert(2,:), 'YData', Insert(1,:)); drawnow; 
        end
    end

    % Removal loop
    remove = 1;
    while remove ~= 0
        figure(gcf); remove = input('Remove a PulseStart? No(0), or Yes (Input frame number): ');
        if remove ~= 0
            Insert(:, Insert(2,:) == remove) = [];
            set(hPulseStarts, 'XData', Insert(2,:), 'YData', Insert(1,:)); drawnow; 
        end
    end

    locsA = sort(Insert(2,:)); numPulses = length(locsA) - 1; 
    close(gcf); clear add addX addY remove Insert; clc;

% Scan Other Flow Profiles for Pulses Starts Near Arterial Pulse Starts
    scanWindow = round(0.1 * (FPS / Harmonics(2,1))); %Scans +/-10% of PI, consider adjusting to search only right
    locsG = NaN(size(locsA)); locsB = NaN(size(locsA)); locsV = NaN(size(locsA));

    for i = 1:length(locsA)
    % Define the window around the current location
        startIdx = max(1, locsA(i) - scanWindow);
        endIdx = min(length(FlowProfiles(1,:)), locsA(i) + scanWindow);
    % Extract the segment
        scanG = FlowProfiles(1,startIdx:endIdx);
        scanB = FlowProfiles(2,startIdx:endIdx);
        scanV = FlowProfiles(4,startIdx:endIdx); 
    % Find local minima in each segment
        [~, minLocG] = min(scanG); locsG(i) = startIdx + minLocG - 1;
        [~, minLocB] = min(scanB); locsB(i) = startIdx + minLocB - 1;  
        [~, minLocV] = min(scanV); locsV(i) = startIdx + minLocV - 1;      
    end
    locsG = sort(locsG); locsB = sort(locsB); locsV = sort(locsV);
    PulseStarts = {locsG,locsB,locsA,locsV};

    % Calculate Pulse Timing Information
    PulseInterval = median(locsG(2:end) - locsG(1:end-1)) / FPS; %PI in seconds
    PulseLength = round(PulseInterval * FPS); %PI in frames
    PulseDelay = mean(locsV - locsA) / FPS; %Delay between arterial and venous pulses

    % Automatic Pulse Outlier Thresholding
    xThresh = [prctile((locsG(2:end) - locsG(1:end-1)), 25) - (1.5*iqr(locsG(2:end) - locsG(1:end-1))),... 
        prctile((locsG(2:end) - locsG(1:end-1)), 75) + (1.5*iqr(locsG(2:end) - locsG(1:end-1)))]; 
    for j = 1:4
        yThresh(j,1) = prctile(FlowProfiles(j, :), 10) - (1.5*iqr(FlowProfiles(j,:)));
        yThresh(j,2) = prctile(FlowProfiles(j, :), 90) + (1.5*iqr(FlowProfiles(j,:)));
    end

    clear thresholdPI pksA startIdx endIdx scanG scanB scanV minLocG minLocB minLocV i scanWindow locsG locsB locsA locsV; clc;

% Pulse Exclusions (With Manual Corroboration)
    %Plot Pulse Start Candidates 
    close all;
    figure;
    for i = 1:4
        subplot(4, 1, i);
        plot(TimeFlow, FlowProfiles(i, :), colors{i}, 'LineWidth', 1.5);
        hold on;
        plot(TimeFlow(PulseStarts{i}), FlowProfiles(i, PulseStarts{i}), 'ko');
        for j = 1:length(PulseStarts{i})
            text(TimeFlow(PulseStarts{i}(j)), FlowProfiles(i, PulseStarts{i}(j)), num2str(j), ...
                'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'FontSize', 8, 'Color', 'r');
        end
        title(strcat(titles{i}, ' With Pulse Starts'));
    end  

    % Load Pulses
    Pulses = cell(1, 4);
    for j = 1:4
        InsertProfile = nan(numPulses, max(PulseStarts{j}(2:end) - PulseStarts{j}(1:end-1)+1));
        for i = 1:numPulses
            startIdx = PulseStarts{j}(i);
            endIdx = PulseStarts{j}(i+1);
            if endIdx <= length(FlowProfiles(j, :))
                InsertPulse = FlowProfiles(j, startIdx:endIdx);  
            else; InsertPulse = FlowProfiles(j, startIdx:end); end;

            % Exclude outliers
            Toggle = 0;
            if Toggle == 1; 
                if any(InsertPulse<yThresh(j,1)) || any(InsertPulse>yThresh(j,2)) == 1
                    InsertPulse(:) = NaN;
                elseif length(InsertPulse)<xThresh(1) || length(InsertPulse)>xThresh(2) == 1
                    InsertPulse(:) = NaN;
                end
            end
            InsertProfile(i, 1:length(InsertPulse)) = InsertPulse;
        end
        Pulses{j} = InsertProfile;
    end

    % Nonperiodic Exclusion
    PeriodicPulses = Pulses;
    PeriodicRange = [0.90*PulseLength, 1.1*PulseLength];

% Pulse Exclusion GUI
    % GUI NOTE: right click pulse & show property editor for pulse #
    
    clc;  disp(['Exclude Breathes and pulses that are too long/short']); 
    figure;
    hold on;
        remove = 1;
        while remove ~= 0
            clf; 
            for j = 1:4
                subplot(2, 2, j); hold on;              
                Extract = PeriodicPulses{j};

                % Plot individual pulse traces with labels
                for i = 1:size(Extract, 1)
                    h = plot(Extract(i, :), 'k:', 'LineWidth', 1);
                    lastIdx = find(~isnan(Extract(i, :)), 1, 'last'); 
                    if ~isempty(lastIdx)
                        text(lastIdx, Extract(i, lastIdx), num2str(i), 'FontSize', 10, 'FontWeight', 'bold');
                    end
                l1 = legend; set(l1,'Visible','off');
                end

                line([PeriodicRange(1) PeriodicRange(1)], ylim, 'Color', 'r', 'LineWidth', 3);
                line([PeriodicRange(2) PeriodicRange(2)], ylim, 'Color', 'r', 'LineWidth', 3);
                line([PulseLength PulseLength], ylim, 'Color', 'g', 'LineWidth', 3);

                % Compute and plot mean waveform and SNR    
                AvgFlowProfiles{j} = mean(Extract, 1, 'omitnan');   
                line(xlim, [AvgFlowProfiles{j}(1) AvgFlowProfiles{j}(1)], 'Color', 'g', 'LineWidth', 3);

                hPlots(j) = plot(AvgFlowProfiles{j}, 'LineWidth', 3,'Color','k'); 
                PulseMean = nanmean(PeriodicPulses{j},1); PulseSTD = nanstd(PeriodicPulses{j},0,1); 
                Signal = range(PulseMean); Noise = nanmean(PulseSTD); 
                SNRdb = 10 * log10(Signal / Noise); StoreSNRdb(j) = SNRdb;

                textX = xlim; textX = textX(2)*0.975; textY = ylim; textY = textY(2)*0.975;
                text(textX,textY, ['SNR = ', num2str(SNRdb, '%.2f'), ' dB'], 'HorizontalAlignment', 'left', 'FontSize', 12, 'FontWeight', 'bold','Color','red');
                title(titles{j});
            end
            disp(['Max SNR is ', num2str(max(StoreSNRdb)), ' dB']); 

            % Ask user for pulse removal
            remove = input('Remove a Pulse? (If no, enter 0, otherwise indicate pulse number): ');
            if remove ~= 0
                for j = 1:4; PeriodicPulses{j}(remove, :) = NaN; end;
            end
        end
    hold off;

    clear h hPlots Extract l1 textX textY remove;

% Temporal Standardization and Quality Control
    InterpolatedPulses = cell(size(Pulses));
    standardLength = 1000;
    for j = 1:numel(PeriodicPulses)
        numPulses = size(PeriodicPulses{j}, 1); % Get number of pulses in current profile
        InterpolatedPulses{j} = NaN(numPulses, standardLength); % Preallocate for the interpolated pulses

        for i = 1:numPulses
            extractPulse = PeriodicPulses{j}(i, :); % Extract the i-th pulse

            % Remove NaNs from pulse
            validIdx = ~isnan(extractPulse); % Index to find valid data (no NaNs)
            validPulse = extractPulse(validIdx); % Create a valid pulse array

            if isempty(validPulse)
                continue; % Skip if pulse is empty after NaN removal
            end

            originalLength = length(validPulse); % Determine length of the valid pulse
            originalTime = linspace(0, 1, originalLength); % Define original time vector
            newTime = linspace(0, 1, standardLength); % Define new standardized time vector

            % Interpolate the pulse to the new time bins using 'pchip'
            InterpolatedPulses{j}(i, :) = interp1(originalTime, validPulse, newTime, 'pchip', 'extrap');
        end
    end
    InterpSmoothWindow = round(standardLength/PulseLength);

% Pulse Exclusion GUI
clc; close(figure(2));  disp(['Exclude Nonperiodic Pulses']); 
figure;
hold on;
    remove = 1;
    while remove ~= 0
        clf; 
        for j = 1:4
            subplot(2, 2, j); hold on;        
            Extract = InterpolatedPulses{j};
            
            % Plot individual pulse traces with labels
            for i = 1:size(Extract, 1)
                h = plot(Extract(i, :), 'k:', 'LineWidth', 1);
                lastIdx = find(~isnan(Extract(i, :)), 1, 'last'); 
                if ~isempty(lastIdx)
                    text(lastIdx, Extract(i, lastIdx), num2str(i), 'FontSize', 10, 'FontWeight', 'bold');
                end
            l1 = legend; set(l1,'Visible','off');
            end
            
            % Compute and plot mean waveform and SNR 
            AvgFlowProfiles{j} = mean(Extract, 1, 'omitnan');      
            line(xlim, [AvgFlowProfiles{j}(1) AvgFlowProfiles{j}(1)], 'Color', 'g', 'LineWidth', 3);

            hPlots(j) = plot(AvgFlowProfiles{j}, 'LineWidth', 3,'Color','k'); 
            PulseMean = nanmean(InterpolatedPulses{j},1); PulseSTD = nanstd(InterpolatedPulses{j},0,1); 
            Signal = range(PulseMean); Noise = nanmean(PulseSTD); 
            SNRdb = 10 * log10(Signal / Noise); StoreSNRdb(j) = SNRdb;
            
            textX = xlim; textX = textX(2)*0.975; textY = ylim; textY = textY(2)*0.975;
            text(textX,textY, ['SNR = ', num2str(SNRdb, '%.2f'), ' dB'], 'HorizontalAlignment', 'left', 'FontSize', 12, 'FontWeight', 'bold','Color','red');
            title(titles{j});
        end
        disp(['Max SNR is ', num2str(max(StoreSNRdb)), ' dB']); 
        
        % Ask user for pulse removal
        remove = input('Remove a Pulse? (If no, enter 0, otherwise indicate pulse number): ');
        if remove ~= 0
            for j = 1:4; InterpolatedPulses{j}(remove, :) = NaN; end;
        end
    end
hold off;

if Save == 1
    saveas(gcf, strcat(DataName, " Selected Pulses.tif"));
    saveas(gcf, strcat(DataName, " Selected Pulses.fig"));
end 
 
clear remove xlim ylim textX textY SNRdb Signal Noise lastIdx InsertPulse InsertProfile 
clear validIdx selectedPulses endIdx startIdx Extract l1 i j h hPlots hPulseStarts hWaveform
clc;

%% SECTION 6: Hemodynamic Analysis
%Updating Pulses and PI
SelectedPulses = Pulses;
    for j = 1:numel(Pulses)
        selectedPulses= nan(size(InterpolatedPulses{j},1),1);
        validIdx = ~isnan(InterpolatedPulses{j}(:,1)); 
        selectedPulses(validIdx) = 1; 
        SelectedPulses{j}(~validIdx,:) = NaN;
        PulseInterval2(j) = (sum(sum(~isnan(SelectedPulses{j}),2)))/(FPS*sum(any(~isnan(SelectedPulses{j}), 2)));
        BasalFlow(j) = mean(SelectedPulses{j}(:,1),'omitnan');
        AvgFlowProfiles{j} = AvgFlowProfiles{j}(1,:)-BasalFlow(j);
        %Peaks{j} = Peaks{j}(1,:)-BasalFlow(j);
        %Notches{j} = Notches{j}(1,:)-BasalFlow(j);
    end   

%Preliminary Automatic Peak Detection
nPeaks = 3; Peaks = cell(size(AvgFlowProfiles, 2), 1);
if exist('standardLength','var') == 1
    PulseLength = 1000;
end

for i = 1:size(AvgFlowProfiles, 2)
    % Indexing Peaks
    [PKS, LOCS, W, P] = findpeaks(AvgFlowProfiles{i}); % Find maxima (peaks)
        [~, peakOrder] = sort(W, 'descend'); % Sort peaks by width
        selectedPeaks = peakOrder(1:min(nPeaks, length(peakOrder)));
        selectedPeaks = sort(selectedPeaks);
        selectedPeaks = selectedPeaks(LOCS(selectedPeaks)<PulseLength);
    Peaks{i} = [PKS(selectedPeaks); LOCS(selectedPeaks)]; % Store peak values and locations
end

% Derivative Subplotting & Manual Peak Corrobroation/Addition
figure
hold on
    add = 1; 
    while add ~=0   
        for i = 1:size(AvgFlowProfiles,2)
        subplot(3,4,i)
        hold on
            plot(AvgFlowProfiles{i}(1:PulseLength),colors{i});
            plot(Peaks{i}(2, :),Peaks{i}(1, :), 'v', 'MarkerSize', 5, 'MarkerFaceColor', colors{i}, 'MarkerEdgeColor', 'k');
            title(titles{i}); if i ==1; ylabel('Flow, 1/K^2 (a.u.)'); end;  
        subplot(3,4,(i+4))
            Deriv1{i} = diff(AvgFlowProfiles{i}(1:PulseLength));
            Deriv1{i} = movmean(Deriv1{i}, InterpSmoothWindow, 'Endpoints', 'shrink');
            plot(Deriv1{i},colors{i}); if i ==1; ylabel('1st Derivative'); end;
        subplot(3,4,(i+8))
            Deriv2{i} = diff(AvgFlowProfiles{i}(1:PulseLength),2);    
            Deriv2{i} = movmean(Deriv2{i}, InterpSmoothWindow, 'Endpoints', 'shrink');
            plot(Deriv2{i},colors{i}); if i ==1; ylabel('2nd Derivative'); end;
        end
    add = input('Add a hidden peak? No(0), Global(1), Background(2), Arterial(3), or Venous(4): ');
    figure(gcf); 
        if add ~= 0
            frame = input('What frame?: '); figure(gcf);
            insert = [AvgFlowProfiles{add}(frame);frame];
            Peaks{add} = [Peaks{add} insert];
        end
        for i = 1:numel(Peaks)
            extract = Peaks{i}; 
            [~, update] = sort(extract(2, :));
            Peaks{i} = extract(:, update);
        end    
    end

    % Automatic Notch Detection
    Notches = cell(size(AvgFlowProfiles, 2), 1);
    for i = 1:size(AvgFlowProfiles, 2)        
        % Skip loop if no notches
        nPeaks = size(Peaks{i},2);
        if size(Peaks{i},2) < 2; 
            Notches{i} = []; continue;
        end

        % Indexing Notches
        [NPKS, NLOCS, W, P] = findpeaks(-AvgFlowProfiles{i}); % Find minima (notches)
        NPKS = -NPKS;

        selectedNotches = [];
        for k = 1:(size(Peaks{i},2)-1)
            % Get notches between Peak k and Peak k+1
            inRange = (NLOCS >Peaks{i}(2,k) & NLOCS < Peaks{i}(2,(k+1)));
            validNLOCS = NLOCS(inRange); % Locations of candidate notches
            validNPKS = NPKS(inRange); % Corresponding values
            if ~isempty(validNLOCS)
                [deepestNotch, idx] = min(validNPKS); % Select deepest notch
                selectedNotches = [selectedNotches, [deepestNotch; validNLOCS(idx)]];
            end
        end
        Notches{i} = selectedNotches; % Store only valid notches
        
        if isempty(Notches{i}) == 1; continue; end;
        subplot(3,4,i)
        hold on
            plot(Notches{i}(2, :),Notches{i}(1, :), 'v', 'MarkerSize', 5, 'MarkerFaceColor', colors{i}, 'MarkerEdgeColor', 'k');
            title(titles{i}); if i ==1; ylabel('Flow, 1/K^2 (a.u.)'); end;  
    end
hold off; 
figure(gcf);

if Save == 1
    saveas(gcf, strcat(DataName, " Derivatives.tif"));
    saveas(gcf, strcat(DataName, " Derivatives.fig"));
end  

clear PKS LOCS W P peakOrder selectedPeaks notchOrder Deriv1 Deriv add frame insert update;
clear nPeaks NPKS NLOCS selectedNotches extract validNLOCS validNPKS deepestNotch j i idx ; clc;

%% CHECKPOINT
if Save == 1
    save(strcat(DataName,".mat")); display('Workplace has been saved as a checkpoint');
end     

%% OUTPUT PLOTS: Plot Pulse Detection and Waveform Averages
% Arterial & Venous PSD's
figure
    hold on
        plot(FreqPSD(1:PSDmidpoint),LinearPSD(1,1:PSDmidpoint),'g','LineWidth',3,'DisplayName', 'Global Flow');
        plot(FreqPSD(1:PSDmidpoint),LinearPSD(2,1:PSDmidpoint),'k','LineWidth',3,'DisplayName', 'Background Flow');
        plot(FreqPSD(1:PSDmidpoint),mean(ArterialPSD(:,1:PSDmidpoint),'omitnan'),'r','LineWidth',3,'DisplayName', 'Arterial Flow');
        plot(FreqPSD(1:PSDmidpoint),mean(VenousPSD(:,1:PSDmidpoint),'omitnan'),'b','LineWidth',3,'DisplayName', 'Venous Flow');
        l1 = legend; set(l1, 'FontSize', 10, 'Location', 'NorthEast');
        xlabel('Frequency (Hz)'); ylabel('Power (dB)');
    hold off
if Save == 1
    saveas(gcf, strcat(DataName, " Flow Profile PSDs.tif"));
    saveas(gcf, strcat(DataName, " Flow Profile PSDs.fig"));
end  
    
% Flow Profiles Annotated with Pulse Starts
figure;
for i = 1:4
    subplot(4, 1, i);
        plot(TimeFlow, FlowProfiles(i, :),colors{i},'LineWidth',1.5);
        hold on;
        plot(TimeFlow(PulseStarts{i}), FlowProfiles(i, PulseStarts{i}), 'ko');  
            validIdx = ~isnan(SelectedPulses{i}(:,1));
            validStart = PulseStarts{i}(validIdx); 
        plot(TimeFlow(validStart), FlowProfiles(i, validStart), '.', 'MarkerSize', 15, 'Color', colors{i});
        title(strcat(titles{i},' With Pulse Starts'))
end
hold off
if Save == 1
    saveas(gcf, strcat(DataName, " Pulse Starts.tif"));
    saveas(gcf, strcat(DataName, " Pulse Starts.fig"));
end  

% Baselined Average Flow Profiles with Major Features
TimeWaveform = TimeFlow(1:PulseLength);
figure;
    hold on;
    for i = 1:4
        TimeWaveform = (1:size(AvgFlowProfiles{i},2))*PulseInterval2(i);
        h{i} = plot(TimeWaveform, AvgFlowProfiles{i}, colors{i}, 'LineWidth', 3, 'DisplayName', titles{i});
        plot(TimeWaveform(Peaks{i}(2, :)), Peaks{i}(1, :), 'v', 'MarkerSize', 8, 'MarkerFaceColor', colors{i}, 'MarkerEdgeColor', 'k');
        if isempty(Notches{i}) == 1; continue; end;
        plot(TimeWaveform(Notches{i}(2, :)), Notches{i}(1, :), '^', 'MarkerSize', 8, 'MarkerFaceColor', colors{i}, 'MarkerEdgeColor', 'k');
    end
        title('Flow Profiles'); xlabel('Time (ms)'); ylabel('Flow, 1/K^2 (a.u.)');
        legend([h{:}])
    hold off;
if Save == 1
    saveas(gcf, strcat(DataName, " Average Waveforms.tif"));
    saveas(gcf, strcat(DataName, " Average Waveforms.fig"));
end     

% % Pulse Traces with SNR
% figure;
% for j = 1:4
%     subplot(2, 2, j);
%     hold on;
%     % Truncate or pad pulses
%         AdjPulses = NaN(length(Pulses{j}), PulseLength);
%         for i = 1:length(Pulses{j})
%             ExtractPulse = Pulses{j}{i};
%                 if length(ExtractPulse) >= PulseLength
%                     AdjPulses(i, :) = ExtractPulse(1:PulseLength);  
%                 else; AdjPulses(i, 1:length(ExtractPulse)) = ExtractPulse; end
%             TimePulse = (0:PulseLength-1) / FPS;
%             plot(TimePulse, AdjPulses(i, :), 'k:');  
%         end
%         ValidPeaks = Peaks{j}(2, :) <= PulseLength; 
%         ValidNotches = Notches{j}(2, :) <= PulseLength;
%     % Calculate and display SNR for each Flow Profile
%         PulseMean = nanmean(AdjPulses,1);  
%         PulseSTD = nanstd(AdjPulses,0,1); 
%         Signal = max(PulseMean) - min(PulseMean); 
%         Noise = nanmean(PulseSTD); 
%         SNRdb = 10 * log10(Signal / Noise); StoreSNRdb(j) = SNRdb;
%         disp(['SNR for ', titles{j}, ' = ', num2str(SNRdb), ' dB']);   
%     % Plot Pulse Variance with SNR
%         plot(TimeWaveform(1:PulseLength), PulseMean, colors{j}, 'LineWidth', 3, 'DisplayName', titles{j});
%         plot(TimeWaveform(Peaks{j}(2, ValidPeaks)), Peaks{j}(1, ValidPeaks), 'v', 'MarkerSize', 8, 'MarkerFaceColor', colors{j}, 'MarkerEdgeColor', 'k');
%         plot(TimeWaveform(Notches{j}(2, ValidNotches)), Notches{j}(1, ValidNotches), '^', 'MarkerSize', 8, 'MarkerFaceColor', colors{j}, 'MarkerEdgeColor', 'k');
%             textX = xlim; textX = textX(2)*0.975; textY = ylim; textY = textY(2)*0.975
%         text(textX,textY, ['SNR = ', num2str(SNRdb, '%.2f'), ' dB'], 'HorizontalAlignment', 'right', 'FontSize', 8, 'FontWeight', 'bold');
%         title(['Average ' titles{j} ' Pulse']); xlabel('Time (s)'); ylabel('Flow, 1/K^2 (a.u.)');
%     hold off;
% end
% if Save == 1
%     saveas(gcf, strcat(DataName, " Pulse Traces w SNR.tif"));
%     saveas(gcf, strcat(DataName, " Pulse Traces w SNR.fig"));
% end 
% % deltaSNR = (10^((StoreSNRdb(3)-StoreSNRdb(4))/10)-1)*100 

clear AdjPulses ExtractPulse TimePulse PulseMean PulseSTD Signal Noise SNRdb ValidPeaks ValidNotches h i j TimePulse; clc; 


%% Hemodynamics
Summary = [];
for i = 1:numel(Pulses)
% Extract current flow profile and major features
    waveform = AvgFlowProfiles{i}; 
    peaks = Peaks{i}; notches = Notches{i};
    nPeaks = size(peaks, 2); nNotches = size(notches, 2);
    
% Define Non-volumetric Hemodynamic Metrics for each flow profile   
    % Overall metrics
    MeanFlow = mean(waveform,'omitnan')+BasalFlow(i); 
    MaxFlow = max(waveform)+BasalFlow(i);
    
    % Calculate Peak Metrics
    peak_mag = nan(3,1); peak_amp = nan(3,1); peak_time = nan(3,1);
    for j = 1:max([nPeaks,2])
        if j > nPeaks; break; end;
        peak_mag(j) = peaks(1,j) + BasalFlow(i); 
        peak_amp(j) = peaks(1,j);
        peak_time(j) = peaks(2,j)/PulseLength; 
    end
    
    % Calculate Notch Metrics
    notch_mag = nan(2,1); notch_amp = nan(2,1); notch_time = nan(2,1);
    for j = 1:max([nNotches,2])
        if j > nNotches; break; end;
        notch_mag(j) = notches(1,j) + BasalFlow(i);
        notch_amp(j) = notches(1,j);
        notch_time(j) = notches(2,j)/PulseLength; 
    end 

% Volumetric Hemodynamic Metrics (via Notches)
    time = linspace(0, PulseInterval2(i), standardLength);
    total_volume = trapz(time, (waveform+BasalFlow(i)));
    total_volume_pulsatile = trapz(time, waveform);
   
    segment_volume_total = nan(3,1); segment_volume_pulsatile = nan(3,1);
    add = 1; 
    for j = 1:nPeaks
        if j <= nNotches
            segment_volume_total(j) = trapz(time(add:notches(2,j)), waveform(add:notches(2,j))+BasalFlow(i));
            segment_volume_pulsatile(j) = trapz(time(add:notches(2,j)), waveform(add:notches(2,j)));
            add = notches(2,j);
        else
            segment_volume_total(j) = trapz(time(add:end), (waveform(add:end)+BasalFlow(i)));
            segment_volume_pulsatile(j) = trapz(time(add:end), waveform(add:end));
        end
    end

%     % Calculate Time Delays
%         peak_gap = peak_time(2:end)-peak_time(1:end-1);
%         notch_gap = peak_time(2:end)-peak_time(1:end-1);
%         time_delay = notch_time(1:end)-peak_time(1:end-1);          
%     % Calculate Peak Mag/Amp Ratios Relative to 1st Peak
%         peak_mag_ratio = peak_mag(1:end)./peak_mag(1);
%         peak_amp_ratio = peak_amp(1:end)./peak_amp(1);
%         peak_time_ratio = peak_time(1:end)./peak_time(1);
%     % Calculate Notch Mag/Amp Ratios Relative to 1st Notch
%         notch_mag_ratio = notch_mag(1:end)./notch_mag(1);
%         notch_amp_ratio = notch_amp(1:end)./notch_amp(1);
%         notch_time_ratio = notch_time(1:end)./notch_time(1);    
    Summary(:,i) = [PulseInterval2(i); PulseDelay; BasalFlow(i); MeanFlow; MaxFlow; peak_mag; notch_mag; peak_time; notch_time; total_volume; segment_volume_total; total_volume_pulsatile; segment_volume_pulsatile];
end

if Save == 1
    save(DataName); display('Saved Workplace has been updated');
end 
display('Export AverageFlowProfiles and PPGsummary');

%% Clean up
clearvars -except Summary; clc;
display('Open summary and copy to clipboard for export to excel')
