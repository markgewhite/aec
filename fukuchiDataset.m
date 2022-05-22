classdef fukuchiDataset < modelDataset
    % Subclass for loading the Fukuchi et al (2018) dataset

    properties
        Category            % type of data 
        YReference          % chosen Y variable
        SubjectID           % array of subject IDs matching X and Y
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

            [ XRaw, Y, S, labels ] = fukuchiDataset.load( set, args );

            % setup padding
            pad.length = 101;
            pad.longest = false;
            pad.location = 'Right';
            pad.value = 0;
            pad.same = true;
            pad.anchoring = 'Both';

            tSpan= linspace( 0, 100, 101 );
        
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

        end

    end

    methods (Static)

        function [X, Y, S, names ] = load( set, args )

            if ismac
                rootpath = '/Users/markgewhite/Google Drive/';
            else 
                rootpath = 'C:\Users\m.g.e.white\My Drive\';
            end

            dataFolder = 'Academia/Postdoc/Datasets/Fukuchi';
            datapath = [ rootpath dataFolder ];

            % load the meta data
            metaData = readtable( fullfile(datapath, 'WBDSInfo.xlsx') );

            metaData.Subject = categorical( metaData.Subject );
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

            if args.FromMatlabFile ...
                    && isfile( fullfile(datapath,'RecentData.mat') )

                % load data from a previous run - this is faster
                load( fullfile(datapath,'RecentData.mat'), ...
                      'X', 'Y', 'S', 'names' );

            else
                % load the trial data from original files
                switch args.Category
                    case 'GRF'
                        [ X, Y, S, names ] = loadGRFData( datapath, ...
                                                    refY, ...
                                                    metaData.Subject, ...
                                                    args );

                    case 'JointAngles'

                    case 'JointMoments'

                end
                        
                % save it for future reference
                save( fullfile(datapath,'RecentData.mat'), ...
                      'X', 'Y', 'S', 'names', '-v7.3' );
            
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

end


function [X, Y, S, side] = loadGRFData( datapath, ref, subject, args )

    % constants
    nSubjects = 42;
    nTrials = 8;
    nFiles = nSubjects*nTrials;
    nMaxCycles = 25;

    % pre-allocate space
    X = cell( nFiles*nMaxCycles, 1 );
    Y = zeros( nFiles*nMaxCycles, 1 );
    S = zeros( nFiles*nMaxCycles, 1 );
    side = zeros( nFiles*nMaxCycles, 1 );

    % identify the required fields
    fields = [];
    if args.HasGRF
        if args.HasVGRFOnly
            fields = 4;
        else
            fields = [ 2 3 4 ];
        end
    end

    if args.HasCOP
        fields = [ fields 5 7 ];
    end

    % read in the data, file by file
    kEnd = 0;
    for i = 1:nFiles

        for j = 1:nTrials

            filename = sprintf( 'WBDS%02uwalkT%02ugrf', i, j );
            trialData = readtable( fullfile(datapath, filename) );
            data{1} = table2array(trialData( :, [1, 2:8] ));
            data{2} = table2array(trialData( :, [1, 9:15] ));

            for m = 1:2

                data{m} = gapFill( data{m} );

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

    loadThreshold = 25;
    minLen = 100;
    
    % identify FZ (files are not consistent)
    maxF = max( data(:,2:4) );
    [~, FZIdx] = max(maxF);
    FZIdx = FZIdx + 1;

    % smooth FZ for reliability
    indicator = smoothdata( data(:, FZIdx), 'Gaussian', 10 );

    % initialize an array to hold up to 25 cycles
    cycles = cell( 25, 1 );

    % find the next swing phase
    t2 = find( indicator < loadThreshold, 1 );

    finished = isempty(t2);
    c = 0;
    while ~finished

        % find the start of the next cycle, the stance phase
        t1 = find( indicator( t2+minLen:end ) > loadThreshold, 1 );
        if isempty( t1 )
            break
        end
        t1 = t1 + t2 + minLen - 1;
 
        % find the end of the stance phase
        t2 = find( indicator( t1+minLen:end ) < loadThreshold, 1 );
        if isempty( t2 )
            break
        end
        t2 = t2 + t1 + minLen - 1;

        % cycle found: store it
        c = c + 1;
        cycles{c} = data( t1:t2, fldIdx );

    end

    cycles = cycles( 1:c );

end


function data = gapFill( data )

    % define the full time span 
    tSpanAll = data( :, 1 );

    % identify the fields for gap filling
    indices = [2:5, 7:8];
    
    for i = indices

        % find rows with non-zero entries (no gaps)
        valid = data(:, i)~=0;

        % define a time span that includes only valid rows
        tSpanValid = data( valid, 1 );
    
        % interpolate in those gaps
        data( :, i ) = interp1(   tSpanValid, ...
                                  data(valid, i), ...
                                  tSpanAll, 'nearest' );

    end

end

