classdef jumpGRFDataset < modelDataset
    % Subclass for loading the countermovement jump VGRF dataset

    properties

    end

    methods

        function self = jumpGRFDataset( set, superArgs )
            % Load the countermovement jump GRF dataset
            arguments
                set        char ...
                    {mustBeMember( set, ...
                                   {'Training', 'Testing'} )}
                superArgs.?modelDataset
            end

            [ XRaw, Y ] = jumpGRFDataset.load( set );
            Y = Y+1;

            % setup padding
            pad.length = 1501;
            pad.longest = false;
            pad.location = 'Left';
            pad.value = 1;
            pad.same = false;
            pad.anchoring = 'Right';

            tSpan= -pad.length+1:0;
        
            % setup fda
            paramsFd.basisOrder = 4;
            paramsFd.penaltyOrder = 2;
            paramsFd.lambda = 1E5; % 1E2
         
            % process the data and complete the initialization
            superArgsCell = namedargs2cell( superArgs );

            self = self@modelDataset( XRaw, Y, tSpan, 'CB', ...
                            superArgsCell{:}, ...
                            padding = pad, ...
                            fda = paramsFd, ...
                            datasetName = "Jumps VGRF Data", ...
                            channelLabels = "VGRF (BW)", ...
                            timeLabel = "Time (ms)", ...
                            channelLimits = [0 2.5] );
            

        end

    end

    methods (Static)

        function [X, Y ] = load( set )

            if ismac
                rootpath = '/Users/markgewhite/Google Drive/';
            else 
                rootpath = 'C:\Users\m.g.e.white\My Drive\';
            end

            dataFolder = 'Academia/Postdoc/Datasets/Jumps';
            datapath = [ rootpath dataFolder ];
            filename = ['jumpGRFData-' set '.mat'];
            
            % load data from file
            load( fullfile( datapath, filename ), ...
                  'grf', 'bwall', 'sDataID', 'sJumpID', ...
                  'jumpOrder', 'nJumpsPerSubject' );
            
            % exclude jumps from specified subject (injury pattern) 
            subjectExclusions = find( ismember( sDataID, 100 ) );
            
            % exclude specific jumps with excessive double movements
            jumpExclusions = [  6104, 6114, 6116, ...
                                9404, 9411, 9413, 9416, ...
                                0101, 0103, 0106, 0111, 0114 ];
            
            % set the options for jump detection
            options.tFreq = 1; % time intervals per second
            options.initial = 1; % initial padding value
            options.threshold1 = 0.08; % fraction of BW for first detection
            options.threshold2 = 0.025; % fraction of BW for sustained low threshold
            options.prctileLimit = 90; % no outliers are beyond this limit
            
            [ rawDataSet, ~, typeSet ] =  extractVGRFData( ... 
                                                grf, bwall, nJumpsPerSubject, ...
                                                sDataID, sJumpID, jumpOrder, ...
                                                subjectExclusions, jumpExclusions, ...
                                                options );
            % extract the relevant data
            X = rawDataSet{2}; % jumps with and without arm swing
            Y = typeSet{2}; % jump class (with/without arm swing)
            
       end

   end


end


function [ curveSet, IDSet, typeSet ] =  extractVGRFData( ...
                                    grf, bwall, nJumpsPerSubject, ...
                                    sDataID, sJumpID, jumpOrder, ...
                                    subjectExclusions, jumpExclusions, ...
                                    setup )

    % Extract vertical jumps from the first data collection
    % Data from both force plates have already been added together
    % and the only the vertical component retained.

    visualCheck = false;
    
    nSubjects = length( sDataID );
    
    nTotal = sum( nJumpsPerSubject );
    
    % VGRF data (vertical jumps only)
    vgrfData = cell( nTotal, 1 );
    vgrfRef = zeros( nTotal, 2 );
    withArms = false( nTotal, 1 );

    k = 0;
    c = 0;
    for i = 1:nSubjects
        
        for j = 1:nJumpsPerSubject(i)
            
            jump = jumpOrder( sJumpID==sDataID(i), j );
            jumpID = sDataID(i)*100+j;
    
            if jump{1}(1) == 'V' ...
                    && ~ismember( i, subjectExclusions ) ...
                    && ~ismember( jumpID, jumpExclusions )
    
                c = c+1;
                % find jump initiation and jump take-off
                [tStart, tEnd, valid] = demarcateJump( grf.raw{i,j}, ...
                                                bwall(i,j), ...
                                                setup.threshold1, ...
                                                setup.threshold2 );
    
                if valid
    
                    if visualCheck && i==42
                       plot( grf.raw{i,j}( tStart:tEnd )/bwall(i,j) );
                       disp( num2str(jumpID) );
                       pause;
                    end
    
                    k = k+1;
                
                    % store the VGRF data in bodyweight units
                    vgrfData{ k } = grf.raw{i,j}( tStart:tEnd )/bwall(i,j);
        
                    vgrfRef( k, : ) = [ i j ];
                    withArms( k ) = (length(jump{1}) == 2);
       
                end
                    
            end
    
    
            
        end
    end
    
    % trim the arrays
    vgrfData = vgrfData(1:k);
    vgrfRef = vgrfRef(1:k,:);
    withArms = withArms(1:k,:);
    
    % check lengths
    len = cellfun( @length, vgrfData(1:k) );  
    outliers = len > prctile( len, setup.prctileLimit );
    
    % exclude outliers 
    vgrfData = vgrfData(~outliers);
    vgrfRef = vgrfRef(~outliers,:);
    withArms = withArms(~outliers,:);
    
    % combine all datasets together
    curveSet = { vgrfData(~withArms), vgrfData };
    IDSet = { vgrfRef(~withArms,:), vgrfRef };
    typeSet = { false(length(IDSet{1}),1), withArms };

end


function [ t1, t2, valid ] = demarcateJump( vgrf, bw, threshold1, threshold2 )
% Identify the start and end times of a jump by examining the vertical GRF
% Start point is when the vGRF departs from the mean value in the first
% second by 5 times the standard deviation. 
% End point is when the vGRF reaches (< 10 N)

    % constants
    mingrf = 10; % Newtons
    
    % find take-off - the demarcation end point
    t2 = find( abs(vgrf) < mingrf, 1 );
    
    % normalise to bodyweights
    vgrf = smoothdata( (vgrf-bw)/bw, 'Gaussian', 21 );
    
    % detect the first significant movement
    t3 = find( abs(vgrf) > threshold1, 1 );
    if isempty( t3 )
        valid = false;
        return
    end
    
    % work backwards to find where vGRF falls to < lower threshold
    % ensure this is a stable low period 
    t1 = t3;
    while t1>1  && t3>1
        % find the next point backwards where VGRF is within threshold
        t1 = find( abs(vgrf(1:t3)) < threshold2, 1, 'Last' );
        if isempty(t1)
            t1 = 1;
        end
    
        % from there, find where the VGRF rises back above this threshold
        t3 = find( abs(vgrf(1:t1)) > threshold2, 1, 'Last' );
        if isempty(t3)
            t3 = 1;
        end
    
    end
    
    if isempty(t1)
        t1 = 1;
    else
        % find the last zero crossover point, nearest take-off
        crossover = vgrf(1:t1).*vgrf(2:t1+1);
        t0 = find( crossover < 0, 1, 'last' );
        if isempty(t0)
            % if it does not pass through zero, find the point nearest zero
            [~, t0 ] = min( abs(vgrf(1:t1)) );
        end
        t1 = max( t0 + t3 - 1, 1 );
    end
    
    valid = true;
   
end

