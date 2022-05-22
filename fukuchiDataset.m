classdef fukuchiDataset < modelDataset
    % Subclass for loading the Fukuchi et al (2018) dataset

    properties
        Category            % type of data 
        YReference          % chosen Y variable
        SubjectID           % array of subject IDs matching X and Y
        Side                % indicating left or right
        HasGRF              % whether data includes ground reaction force
        HasVGRFOnly         % whether GRF data has only vertical axis force
        HasCOP              % whether data includes centre of pressure
        FromMatlabFile      % if from Matlab file rather than original data files
    end

    methods

        function self = fukuchiDataset( set, args, superArgs )
            % Load the data
            arguments
                set                 char ...
                    {mustBeMember( set, ...
                    {'Training', 'Testing'} )}
                args.RandomSeed     double = 1234
                args.YReference     char ...
                    {mustBeMember( args.YReference, ...
                    {'AgeGroup', 'Gender', 'SpeedClass', 'GaitSpeed'} )} = 'AgeGroup'
                args.Category       char ...
                    {mustBeMember( args.Category, ...
                    {'GRF', 'JointAngles', 'JointMoments'} )} = 'GRF'
                args.HasGRF         logical = true
                args.HasVGRFOnly    logical = true
                args.HasCOP         logical = false
                args.FromMatlabFile logical = true
                superArgs.?modelDataset
            end

            if (args.HasGRF + args.HasCOP)==0
                eid = 'gaitrecDataset:SignalsNotSpecified';
                msg = 'Neither GRF nor COP signals have been specified.';
                throwAsCaller( MException(eid,msg) );
            end

            [ XRaw, Y, S, side, labels ] = fukuchiDataset.load( set, args );

            % setup padding
            maxLen = max( cellfun( @length, XRaw ) );
            pad.length = maxLen;
            pad.longest = true;
            pad.location = 'Right';
            pad.value = 0;
            pad.same = true;
            pad.anchoring = 'Both';

            tSpan= (0:maxLen-1)/300;
        
            % setup fda
            paramsFd.basisOrder = 4;
            paramsFd.penaltyOrder = 2;
            paramsFd.lambda = 1E2; % 1E2
         
            % process the data and complete the initialization
            superArgsCell = namedargs2cell( superArgs );

            self = self@modelDataset( XRaw, Y, tSpan, 'SBC', ...
                            superArgsCell{:}, ...
                            padding = pad, ...
                            fda = paramsFd, ...
                            datasetName = "Fukuchi", ...
                            channelLabels = labels, ...
                            timeLabel = "Percentage Stance", ...
                            channelLimits = [] );

            self.Category = args.Category;
            self.YReference = args.YReference;
            self.HasGRF = args.HasGRF;
            self.HasCOP = args.HasCOP;
            self.FromMatlabFile = args.FromMatlabFile;

            self.SubjectID = S;
            self.Side = side;

        end

    end

    methods (Static)

        function [X, Y, S, side, names ] = load( set, args )

            if ismac
                rootpath = '/Users/markgewhite/Google Drive/';
            else 
                rootpath = 'C:\Users\m.g.e.white\My Drive\';
            end

            dataFolder = 'Academia/Postdoc/Datasets/Fukuchi';
            datapath = [ rootpath dataFolder ];

            % load the meta data
            metaData = readtable( fullfile(datapath, 'WBDSInfo.xlsx') );

            metaData.AgeGroup = categorical( metaData.AgeGroup );
            metaData.Gender = categorical( metaData.Gender );
            metaData.Dominance = categorical( metaData.Dominance );
            metaData.TreadHands = categorical( metaData.TreadHands );

            % set and apply the filter based on the arguments
            filter = setFilter( metaData, set, args );
            metaData = metaData( filter, : );

            % set Y reference class based on the requested grouping
            switch args.YReference
                case 'AgeGroup'
                    young = metaData.AgeGroup=='Young';
                    refY( young ) = 1;
                    refY( ~young ) = 2;

                case 'Gender'
                    male = metaData.Gender=='M';
                    refY( male ) = 1;
                    refY( ~male ) = 2;

                case 'SpeedClass'
                    metaData.SpeedClass = categorical( metaData.FileName(11:13) );
                    refY = metaData.SpeedClass;

                case 'GaitSpeed'
                    refY = metaData.GaitSpeed;

            end

            filename = ['Recent' set 'Data.mat'];
            if args.FromMatlabFile ...
                    && isfile( fullfile(datapath,filename) )

                % load data from a previous run - this is faster
                load( fullfile(datapath,filename), ...
                      'X', 'Y', 'S', 'side', 'names' );

            else
                % load the trial data from original files
                switch args.Category
                    case 'GRF'
                        [ X, Y, S, side, names ] = loadGRFData( datapath, ...
                                                    refY, ...
                                                    metaData.Subject, ...
                                                    args );

                    case 'JointAngles'

                    case 'JointMoments'

                end
                        
                % save it for future reference
                save( fullfile(datapath,filename), ...
                      'X', 'Y', 'S', 'side', 'names', '-v7.3' );
            
            end
            
       end

   end


end


function filter = setFilter( metaData, set, args )

    % partition the dataset by subject
    subjects = unique( metaData.Subject );
    % select from the 42 subjects with reproducibility
    rng( args.RandomSeed );
    subjectsForTraining = randsample( subjects, 30 );

    % filter by dataset
    switch set
        case 'Training'
            filter = ismember( metaData.Subject, subjectsForTraining );
        case 'Testing'
            filter = ~ismember( metaData.Subject, subjectsForTraining );
    end

    % filter for only treadmill files
    filter = filter & metaData.TreadHands~='--';

    % filter for only txt files
    filter = filter & metaData.FileName(end-2:end)=='txt';

end


function [X, Y, S, side, names] = loadGRFData( datapath, ref, subject, args )

    % constants
    nSubjects = 42;
    nTrials = 8;
    nMaxCyclesPerFile = 60;

    % pre-allocate space
    nCycles = nSubjects*nTrials*nMaxCyclesPerFile;
    X = cell( nCycles, 1 );
    Y = zeros( nCycles, 1 );
    S = zeros( nCycles, 1 );
    side = zeros( nCycles, 1 );

    % identify the required fields
    fields = [];
    names = [];
    if args.HasGRF
        if args.HasVGRFOnly
            fields = 4;
            names = "GRFz";
        else
            fields = [ 2 3 4 ];
            names = ["GRFx", "GRFy", "GRFz"];
        end
    end

    if args.HasCOP
        fields = [ fields 5 7 ];
        names = strcat( names, ["COPx", "COPy"] );
    end

    % read in the data, file by file
    kEnd = 0;
    for i = 1:nSubjects

        for j = 1:nTrials

            filename = sprintf( 'WBDS%02uwalkT%02ugrf', i, j );
            try
                trialData = readtable( fullfile(datapath, filename) );
            catch
                continue
            end
            data{1} = table2array(trialData( :, [1, 2:8] ));
            data{2} = table2array(trialData( :, [1, 9:15] ));

            for m = 1:2

                data{m} = gapFill( data{m}, fields );

                cycleData = extractCycles( data{m}, fields );

                kStart = kEnd + 1;
                kEnd = kStart + length( cycleData ) - 1;

                X( kStart:kEnd ) = cycleData;
                Y( kStart:kEnd ) = ref( (i-1)*nTrials+j );
                S( kStart:kEnd ) = subject( (i-1)*nTrials+j );

                side( kStart:kEnd ) = m;

            end

        end

    end

    X = X( 1:kEnd );
    Y = Y( 1:kEnd );
    S = S( 1:kEnd );
    side = side( 1:kEnd );

end


function cycles = extractCycles ( data, fldIdx )

    loadThreshold = 20;
    minSwingLen = 50;
    minStanceLen = 100;
    offsetStart = 8;
    offsetEnd = 12;
    
    % re-arrange columns into the standard order
    GRF = data( :, 2:4 );
    GRF = GRF( :, [1 3 2] );
    data( :, 2:4 ) = GRF;

    % smooth FZ for reliability
    indicator = smoothdata( data(:, 4), 'Gaussian', 10 );

    % initialize an array to hold up to 25 cycles
    cycles = cell( 25, 1 );

    % find the next swing phase
    t2 = find( indicator < loadThreshold, 1 );

    finished = isempty(t2);
    c = 0;
    while ~finished

        % find the start of the next cycle, the stance phase
        t1 = find( indicator( t2+minSwingLen:end ) > loadThreshold, 1 );
        if isempty( t1 )
            break
        end
        t1 = t1 + t2 + minSwingLen - offsetStart;
        if t1 > length(indicator)
            break
        end
 
        % find the end of the stance phase
        t2 = find( indicator( t1+minStanceLen:end ) < loadThreshold, 1 );
        if isempty( t2 )
            break
        end
        t2 = t2 + t1 + minStanceLen + offsetEnd;
        if t2 > length(indicator)
            break
        end

        % cycle found: store it
        c = c + 1;
        cycles{c} = data( t1:t2, fldIdx );

    end

    cycles = cycles( 1:c );

end


function data = gapFill( data, fldIdx )

    % not dropouts if absolute value is less than threshold
    threshold = 40;

    % define the full time span 
    tSpanAll = data( :, 1 );
    
    for i = fldIdx

        % take moving averages to look for dropouts
        % averaging either side of the point
        m3 = [0; (data(1:end-2,i)+data(3:end,i))/2; 0];

        % find rows with non-zero entries (single gaps)
        valid = data(:, i)~=0 | abs(m3)<threshold;

        % define a time span that includes only valid rows
        tSpanValid = data( valid, 1 );

        % interpolate in those gaps
        data( :, i ) = interp1(   tSpanValid, ...
                                  data(valid, i), ...
                                  tSpanAll, 'pchip' );

    end

end

