classdef GaitrecDataset < ModelDataset
    % Subclass for loading the GaitRec dataset

    properties
        Stratified          % if classes are evenly distributed
        Grouping            % classification scheme
        ShodCondition       % footwear condition
        Speed               % trial speed condition
        Readmission         % whether admission irrespective of number
        SessionType         % either initial, on-going (control), readmission
        SubjectID           % array of subject IDs matching X and Y
        HasGRF              % whether data includes ground reaction force
        HasVGRFOnly         % whether GRF data has only vertical axis force
        HasCOP              % whether data includes centre of pressure
        Side                % left/right selection criterion
        HasDerivative       % whether it includes first derivative of 'side'
        HasDelta            % whether it includes difference between L & R
        FromMatlabFile      % if from Matlab file rather than original data files
    end

    methods

        function self = GaitrecDataset( set, args, superArgs )
            % Load the Gaitrec data set
            arguments
                set                 char ...
                    {mustBeMember( set, ...
                           {'Training', 'Testing', 'Combined'} )}
                args.MaxObs         double ...
                    {mustBePositive, mustBeInteger} = []
                args.Stratified     logical = false
                args.Grouping       char ...
                    {mustBeMember( args.Grouping, ...
                    {'ControlsVsDisorders', 'Disorders'} )} = 'ControlsVsDisorders'
                args.ShodCondition  char ...
                    {mustBeMember( args.ShodCondition, ...
                    {'All', 'Barefoot/Socks', ...
                     'Shoes', 'OrthopedicShoes' } )} = 'Barefoot/Socks'
                args.Speed          char ...
                    {mustBeMember( args.Speed, ...
                    {'All', 'Slow', ...
                     'SelfSelected', 'Fast'} )} = 'SelfSelected'
                args.Readmission    logical = false
                args.SessionType    char ...
                    {mustBeMember( args.SessionType, ...
                    {'All', 'Initial', ...
                     'Control', 'ReadmissionInitial'} )} = 'All'
                args.HasGRF         logical = true
                args.HasVGRFOnly    logical = true
                args.HasCOP         logical = false
                args.Side           char...
                    {mustBeMember( args.Side, ...
                    {'Left', 'Right', 'Random', ...
                     'Affected', 'Unaffected'} )} = 'Affected'
                args.HasDerivative  logical = false
                args.HasDelta       logical = false
                args.FromMatlabFile logical = false
                args.PaddingLength  double = 0
                args.Lambda         double = []
                superArgs.?ModelDataset
            end

            if (args.HasGRF + args.HasCOP)==0
                eid = 'gaitrecDataset:SignalsNotSpecified';
                msg = 'Neither GRF nor COP signals have been specified.';
                throwAsCaller( MException(eid,msg) );
            end

            [ XRaw, Y, S, channelLabels, classLabels ] = loadData( set, args );

            % setup padding
            if args.PaddingLength==0
                pad.Length = 101;
            else
                pad.Length = args.PaddingLength;
            end
            pad.Longest = false;
            pad.Location = 'Left';
            pad.Value = 1;
            pad.Same = false;
            pad.Anchoring = 'Both';

            tSpan= linspace( 0, 100, 101 );
         
            % process the data and complete the initialization
            superArgsCell = namedargs2cell( superArgs );

            self = self@ModelDataset( XRaw, Y, tSpan, ...
                            superArgsCell{:}, ...
                            padding = pad, ...
                            lambda = args.Lambda, ...
                            datasetName = "GaitRec", ...
                            channelLabels = channelLabels, ...
                            timeLabel = "% Stance", ...
                            classLabels = classLabels, ...
                            channelLimits = [] );

            self.Stratified = args.Stratified;
            self.Grouping = args.Grouping;
            self.ShodCondition = args.ShodCondition;
            self.Speed = args.Speed;
            self.Readmission = args.Readmission;
            self.SessionType = args.SessionType;
            self.HasGRF = args.HasGRF;
            self.HasVGRFOnly = args.HasVGRFOnly;
            self.HasCOP = args.HasCOP;
            self.Side = args.Side;
            self.HasDerivative = args.HasDerivative;
            self.HasDelta = args.HasDelta;
            self.FromMatlabFile = args.FromMatlabFile;

            self.SubjectID = S;           

        end


        function unit = getPartitioningUnit( self )
            % Provide the SubjectID for partitioning (overriding parent)
            arguments
                self    GaitrecDataset
            end

            unit = self.SubjectID;

        end

    end

end


function [X, Y, S, channelNames, classNames ] = loadData( set, args )

    path = fileparts( which('GaitrecDataset.m') );
    path = [path '/../../data/gaitrec'];

    % load the meta data
    metaData = readtable( fullfile(path, 'GRF_metadata.csv') );

    metaData.CLASS_LABEL = categorical( metaData.CLASS_LABEL );
    metaData.CLASS_LABEL_DETAILED = ...
                    categorical( metaData.CLASS_LABEL_DETAILED );

    % set and apply the filter based on the arguments
    filter = setFilter( metaData, set, args );
    metaData = metaData( filter, : );

    % set Y reference class based on the requested grouping
    switch args.Grouping
        case 'ControlsVsDisorders'
            controls = metaData.CLASS_LABEL=='HC';
            refY( controls ) = 1;
            refY( ~controls ) = 2;
            classNames = [ "Control", "Disorder" ];

        case 'Disorders'
            controls = metaData.CLASS_LABEL=='HC';
            ankleDisorder = metaData.CLASS_LABEL=='A';
            kneeDisorder = metaData.CLASS_LABEL=='K';
            hipDisorder = metaData.CLASS_LABEL=='H';
            calcaneusDisorder = metaData.CLASS_LABEL=='C';
            refY( controls ) = 1;
            refY( ankleDisorder ) = 2;
            refY( kneeDisorder ) = 3;
            refY( hipDisorder ) = 4;
            refY( calcaneusDisorder ) = 5;
            classNames = [ "Control", ...
                           "Ankle Disorder", ...
                           "Knee Disorder", ...
                           "Hip Disorder", ...
                           "Calcaneus Disorder" ];

    end

    % count the number of files/datasets required
    nFiles = ( args.HasGRF*(3-2*args.HasVGRFOnly) ... 
                        + args.HasCOP*2 );

    % count up the number of channels per trial
    channelsPerTrial = nFiles*(1 + args.HasDerivative + args.HasDelta);

    % initialize arrays, assuming max of 10 trials per session
    nSessions = size( metaData, 1 );
    if nSessions==0
        error('No data with this combination of filters.');
    end
    X = zeros( nSessions*10, 101, channelsPerTrial );

    if args.FromMatlabFile ...
            && isfile( fullfile(path,'RecentData.mat') )

        % load data from a previous run - this is faster
        load( fullfile(path,'RecentData.mat'), ...
              'trialData', 'sessions', 'sources' );

    else
        % load the trial data from original files
        [ trialData, sessions, sources ] = ...
                        loadTrialData( path, nFiles, args );
        % save it for future reference
        save( fullfile(path,'RecentData.mat'), ...
              'trialData', 'sessions', 'sources', '-v7.3' );
    
    end

    subjects = table2array( metaData( :, 1 ) );

    channelNames = strings( channelsPerTrial, 1 );
    c = 0;
    for j = 1:nFiles
        c = c + 1;
        channelNames(c) = strcat( sources(j), "_", args.Side );
        if args.HasDerivative
            c = c + 1;
            channelNames(c) = strcat( sources(j), "_Derivative" );
        end
        if args.HasDelta
            c = c + 1;
            channelNames(c) = strcat( sources(j), "_Delta" );
        end
    end

    % iterate through the sessions
    kStart = 1;
    for i = 1:nSessions

        sessionID = metaData.SESSION_ID(i);

        % determine which side to use, left or right
        sideIdx = getSide( metaData.AFFECTED_SIDE(i), args.Side );

        % identify the trials for this session using the first file
        trials = sessions( :, 1 )==sessionID;
        nSessionTrials = sum( trials );

        % set index range
        kEnd = kStart + nSessionTrials - 1;

        % iterate through the files, adding the trials to array
        c = 0;
        for j = 1:nFiles

            trials = sessions( :, j )==sessionID;
            if sum( trials ) ~= nSessionTrials
                eid = 'gaitrecDataset:TrialsNotConsistent';
                msg = 'The number of trials across files in not consistent.';
                throwAsCaller( MException(eid,msg) );
            end                       

            % store the selected side signal data
            c = c + 1;
            X( kStart:kEnd, :, c ) = trialData( trials, :, sideIdx, j );

            if args.HasDerivative
                % store the derivative for the selected signal
                c = c + 1;
                X( kStart:kEnd, :, c ) = ...
                        trialData( trials, :, 2+sideIdx, j );
            end

            if args.HasDelta
                % store the delta 
                c = c + 1;
                X( kStart:kEnd, :, c ) = ...
                        trialData( trials, :, end, j );
            end

            % store the reference class
            Y( kStart:kEnd ) = refY( i );

            % store the subject ID
            S( kStart:kEnd ) = subjects( i );

        end

        kStart = kEnd + 1;

    end

    % trim back the arrays
    if isempty( args.MaxObs )
        selection = 1:kEnd;
    else
        if args.Stratified
            w = ones( kEnd, 1 );
            for i = 1:length(unique( Y(1:kEnd) ))
                w( Y==i ) = 1/sum( Y==i );
            end
            w = w./sum(w);
        else
            w = ones( kEnd, 1 )/kEnd;
        end
        selection = randsample( kEnd, args.MaxObs, true, w );
    end
    X = X( selection, :, : );
    Y = Y( selection )';
    S = S( selection );
    
    % convert to a cell array
    X = num2cell( X, [2 3] );
    X = cellfun( @(x) permute( x, [2 3 1] ), X, UniformOutput = false );

end


function filter = setFilter( metaData, set, args )

    
    % fiilter by dataset
    switch set
        case 'Training'
            filter = metaData.TRAIN==1;
        case 'Testing'
            filter=  metaData.TEST==1;
        case 'Combined'
            filter= metaData.TRAIN==1 | metaData.TEST==1;
    end


    % filter by shod condition
    switch args.ShodCondition
        case 'Barefoot/Socks'
            filter = filter & metaData.SHOD_CONDITION==1;
        case 'Shoes'
            filter = filter & metaData.SHOD_CONDITION==2;
        case 'OrthopedicShoes'
            filter = filter & metaData.SHOD_CONDITION==3;
    end

    % filter by speed
    switch args.Speed
        case 'Slow'
            filter = filter & metaData.SPEED==1;
        case 'SelfSelected'
            filter = filter & metaData.SPEED==2;          
        case 'Fast'
            filter = filter & metaData.SPEED==3;
    end

    % filter by re-admission
    if ~args.Readmission
        filter = filter & metaData.READMISSION==0;
    end

    % filter by session type
    switch args.SessionType
        case 'Initial'
            filter = filter & metaData.SESSION_TYPE==1;
        case 'Control'
            filter = filter & metaData.SESSION_TYPE==2;          
        case 'ReadmissionInitial'
            filter = filter & metaData.SESSION_TYPE==3;
    end


end


function side = getSide( affected, type )

    switch type

        case 'Affected'
            if affected > 0
                side = affected;
            else
                side = randi(2);
            end

        case 'Left'
            side = 1;

        case 'Right'
            side = 2;

        case 'Random'
            side = randi( 2 );

        case 'Unaffected'
            if affected > 0
                side = 3-affected;
            else
                side = randi(2);
            end

    end

end


function [trialData, sessions, sources] = ...
                                loadTrialData( path, nFiles, args )

    
    dim = nFiles*(1 + 2*args.HasDerivative + args.HasDelta);
    block = 2 + 2*args.HasDerivative + args.HasDelta;

    trialData = zeros( 75732, 101, block, dim );
    sessions = zeros( 75732, dim );
    sources = strings( dim, 1 );

    % only load files required
    k = 0;
    if args.HasGRF

        if ~args.HasVGRFOnly
            k = k + 1;
            sources(k) = "GRF_AP";
            [ trialData( :, :, :, k ), sessions( :, k ) ] = ...
                               importData( path, 'GRF_F', 'AP', ...
                                           args.HasDerivative, ...
                                           args.HasDelta );
            k = k + 1;
            sources(k) = "GRF_ML";
            [ trialData( :, :, :, k ), sessions( :, k ) ] = ...
                               importData( path, 'GRF_F', 'ML', ...
                                           args.HasDerivative, ...
                                           args.HasDelta );
        end

        k = k + 1;
        sources(k) = "GRF_V";
        [ trialData( :, :, :, k ), sessions( :, k ) ] = ...
                           importData( path, 'GRF_F', 'V', ...
                                       args.HasDerivative, ...
                                       args.HasDelta );

    end

    if args.HasCOP

        k = k + 1;
        sources(k) = "COP_AP";
        [ trialData( :, :, :, k ), sessions( :, k ) ] = ...
                           importData( path, 'GRF_COP', 'AP', ...
                                       args.HasDerivative, ...
                                       args.HasDelta );

        k = k + 1;
        sources(k) = "COP_ML";
        [ trialData( :, :, :, k ), sessions( :, k ) ] = ...
                           importData( path, 'GRF_COP', 'ML', ...
                                       args.HasDerivative, ...
                                       args.HasDelta );

    end


end


function [ data, sessions ] = importData( path, type, dir, ...
                                         inclDeriv, inclDelta )

    dim = 2 + 2*inclDeriv + inclDelta;
    data = zeros( 75732, 101, dim );

    fileprefix = [ type '_' dir '_PRO_' ];

    fileTable = readtable( fullfile(path, [fileprefix 'left.csv'] ) );
    fileData = table2array( fileTable );
    data( :, :, 1 ) = fileData( :, 4:104 );
    sessions = fileData( :, 2 ); % assumed to be same for left and right

    fileTable = readtable( fullfile(path, [fileprefix 'right.csv'] ) );
    fileData = table2array( fileTable );
    data( :, :, 2 ) = fileData( :, 4:104 );

    if inclDeriv
        data( :, 1:100, 3 ) = diff( data( :, :, 1 ), 1, 2 );
        data( :, 1:100, 4 ) = diff( data( :, :, 1 ), 1, 2 );
    end

    if inclDelta
        data( :, :, end ) = data( :, :, 2 ) - data( :, :, 1 );
    end

end
