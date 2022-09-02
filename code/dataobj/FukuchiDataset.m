classdef FukuchiDataset < ModelDataset
    % Subclass for loading the Fukuchi et al (2018) dataset

    properties
        Category            % type of data 
        YReference          % chosen Y variable
        SubjectID           % array of subject IDs matching X and Y
        Side                % indicating left or right
        HasGRF              % whether data includes ground reaction force
        HasVGRFOnly         % whether GRF data has only vertical axis force
        HasCOP              % whether data includes centre of pressure
        HasPelvisAngles     % whether data includes pelvis joint angles
        HasHipAngles        % whether data includes hip joint angles
        HasKneeAngles       % whether data includes knee joint angles
        HasAnkleAngles      % whether data includes ankle joint angles
        HasFootAngles       % whether data includes foot joint angles
        FromMatlabFile      % if from Matlab file rather than original data files
    end

    methods

        function self = FukuchiDataset( set, args, superArgs )
            % Load the data
            arguments
                set                     char ...
                    {mustBeMember( set, ...
                    {'Training', 'Testing'} )}
                args.RandomSeed         double = 1234
                args.YReference         char ...
                    {mustBeMember( args.YReference, ...
                    {'AgeGroup', 'Gender', 'SpeedClass', 'GaitSpeed'} )} = 'AgeGroup'
                args.Category           char ...
                    {mustBeMember( args.Category, ...
                    {'GRF', 'JointAngles'} )} = 'GRF'
                args.HasGRF             logical = true
                args.HasVGRFOnly        logical = true
                args.HasCOP             logical = false
                args.HasPelvisAngles    logical = false
                args.HasHipAngles       logical = false
                args.HasKneeAngles      logical = false
                args.HasAnkleAngles     logical = false
                args.HasFootAngles      logical = false
                args.FromMatlabFile     logical = true
                args.PaddingLength      double = 0
                superArgs.?ModelDataset
            end

            if (args.HasGRF + args.HasCOP)==0
                eid = 'gaitrecDataset:SignalsNotSpecified';
                msg = 'Neither GRF nor COP signals have been specified.';
                throwAsCaller( MException(eid,msg) );
            end

            [ XRaw, Y, S, side, channels, classes ] = FukuchiDataset.load( set, args );

            % setup padding
            if args.PaddingLength==0
                maxLen = max( cellfun( @length, XRaw ) );
                pad.Length = maxLen;
            else
                pad.Length = args.PaddingLength;
            end
            pad.Longest = true;
            pad.Location = 'Right';
            pad.Value = 0;
            pad.Same = true;
            pad.Anchoring = 'Both';

            % setup fda
            paramsFd.BasisOrder = 4;
            paramsFd.PenaltyOrder = 2;
            switch args.Category
                case 'GRF'
                    tSpan= (0:pad.Length-1)/300;
                    label = "Stance Time (s)";
                    paramsFd.Lambda = 1E-6;
                case 'JointAngles'
                    tSpan = 0:100;
                    label = "Percentage Stance (%)";
                    paramsFd.Lambda = 1E-4;
            end          
         
            % process the data and complete the initialization
            superArgsCell = namedargs2cell( superArgs );

            self = self@ModelDataset( XRaw, Y, tSpan, ...
                            superArgsCell{:}, ...
                            padding = pad, ...
                            fda = paramsFd, ...
                            datasetName = "Fukuchi", ...
                            channelLabels = channels, ...
                            timeLabel = label, ...
                            classLabels = classes, ...
                            channelLimits = [] );

            self.Category = args.Category;
            self.YReference = args.YReference;
            self.HasGRF = args.HasGRF;
            self.HasCOP = args.HasCOP;
            self.HasPelvisAngles = args.HasPelvisAngles;
            self.HasHipAngles = args.HasHipAngles;
            self.HasKneeAngles = args.HasKneeAngles;
            self.HasAnkleAngles = args.HasAnkleAngles;
            self.HasFootAngles = args.HasFootAngles;
            self.FromMatlabFile = args.FromMatlabFile;

            self.SubjectID = S;
            self.Side = side;

        end


        function unit = getPartitioningUnit( self )
            % Provide the SubjectID for partitioning (overriding parent)
            arguments
                self    FukuchiDataset
            end

            unit = self.SubjectID;

        end

    end

    methods (Static)

        function [X, Y, S, side, names, classLabels ] = load( set, args )

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
                    classLabels = unique( metaData.AgeGroup );

                case 'Gender'
                    male = metaData.Gender=='M';
                    refY( male ) = 1;
                    refY( ~male ) = 2;
                    classLabels = unique( metaData.Gender );

                case 'SpeedClass'
                    metaData.SpeedClass = categorical( metaData.FileName(11:13) );
                    refY = metaData.SpeedClass;
                    classLabels = unique( metaData.SpeedClass );

                case 'GaitSpeed'
                    refY = metaData.GaitSpeed;
                    classLabels = "";

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
                                                    metaData.FileName, ...
                                                    metaData.Subject, ...
                                                    refY, ...
                                                    args );

                    case 'JointAngles'
                        [ X, Y, S, side, names ] = loadAnglesData( datapath, ...
                                                    metaData.FileName, ...
                                                    metaData.Subject, ...
                                                    refY, ...
                                                    args );

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
    filter = filter & contains( metaData.FileName, 'walkT' );

    % filter for only txt files
    % and for the particular data category
    switch args.Category
        case 'GRF'
            identifier = 'grf';
        case 'JointAngles'
            identifier = 'ang';
        case 'JointMoments'
            identifier = 'knt';
    end

    filter = filter & contains( metaData.FileName, ...
                                [identifier '.txt'] );

end


function [X, Y, S, side, names] = loadGRFData( datapath, filenames, ...
                                               subjects, ref, args )

    % constants
    nFiles = length( filenames );
    nMaxCyclesPerFile = 60;

    % pre-allocate space
    nCycles = nFiles*nMaxCyclesPerFile;
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
        names = [names, "COPx", "COPy"];
    end

    % read in the data, file by file
    filenames = string( filenames );
    kEnd = 0;
    for i = 1:nFiles

        try
            trialData = readtable( fullfile(datapath, filenames(i)) );
        catch
            continue
        end
        
        data{1} = table2array(trialData( :, [1, 2:8] ));
        data{2} = table2array(trialData( :, [1, 9:15] ));

        for j = 1:2

            data{j} = gapFill( data{j}, fields );

            cycleData = extractCycles( data{j}, fields );

            kStart = kEnd + 1;
            kEnd = kStart + length( cycleData ) - 1;

            X( kStart:kEnd ) = cycleData;
            Y( kStart:kEnd ) = ref(i);
            S( kStart:kEnd ) = subjects(i);

            side( kStart:kEnd ) = j;

        end

    end

    X = X( 1:kEnd );
    Y = Y( 1:kEnd );
    S = S( 1:kEnd );
    side = side( 1:kEnd );

end


function [X, Y, S, side, names] = loadAnglesData( datapath, filenames, ...
                                               subjects, ref, args )

    % constants
    nFiles = length( filenames );

    % pre-allocate space
    X = cell( 2*nFiles, 1 );
    Y = zeros( 2*nFiles, 1 );
    S = zeros( 2*nFiles, 1 );
    side = zeros( 2*nFiles, 1 );

    % identify the required fields
    fields = [];
    names = [];
    if args.HasPelvisAngles
        fields = [ fields 2 3 4 ];
        names = ["Pelvic Tilt", "Pelvic Obliquity", "Pelvic Rotation"];
    end

    if args.HasHipAngles
        fields = [ fields 5 6 7 ];
        names = [ names, ...
          "Hip Flexion/Extension", "Hip Add/Abduction", "Hip Int/External Rotation"];
    end

    if args.HasKneeAngles
        fields = [ fields 8 9 10 ];
        names = [names, ...
          "Knee Flexion/Extension", "Knee Add/Abduction", "Knee Int/External Rotation"];
    end

    if args.HasAnkleAngles
        fields = [ fields 11 12 13 ];
        names = [names, ...
          "Ankle Dorsi/Plantarflexion", "Ankle Inv/Eversion", "Ankle Add/Abduction"];
    end

    if args.HasFootAngles
        fields = [ fields 14 15 16 ];
        names = [names, ...
          "Foot Dorsi/Plantarflexion", "Foot Inv/Eversion", "Foot Add/Abduction"];
    end

    % read in the data, file by file
    filenames = string( filenames );
    k = 0;
    for i = 1:nFiles

        try
            trialData = readtable( fullfile(datapath, filenames(i)) );
        catch
            continue
        end
        
        data{1} = table2array(trialData( :, [2:4  8:10 14:16 20:22 26:28] ));
        data{2} = table2array(trialData( :, [5:7 11:13 17:19 23:25 29:31] ));

        for j = 1:2

            cycleData = data{j}( :, fields );

            k = k+1;
            X{k} = cycleData;
            Y(k) = ref(i);
            S(k) = subjects(i);

            side(k) = j;

        end

    end

    X = X( 1:k );
    Y = Y( 1:k );
    S = S( 1:k );
    side = side( 1:k );

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

