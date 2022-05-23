classdef exemplarDataset < modelDataset
    % Subclass for generating an exemplar dataset
    % for testing the capabilities of different models

    properties
        FeatureType         % formula type for the feature element
        ClassSizes          % number observations per class (vector)
        ClassElements       % number of elements per class
        ClassMeans          % positions of features
                            % columns are elements, rows are classes
        ClassSDs            % standard deviations of classes
        ClassPeaks          % amplitudes of classes
        MeanCovariance      % within-class position variations
                            % cell array element per class
                            % symmetric, positive-definite array per class
        SDCovariance        % within-class SD variations
        PeakCovariance      % within-class peak variations
        Noise               % noise
        HasVariableLength   % time series have variable
        TerminationValue    % truncate the series at this value, if variable length
    end

    methods

        function self = exemplarDataset( set, args, superArgs )
            % Initialize the exemplar dataset
            arguments
                set                     char ...
                        {mustBeMember( set, ...
                                   {'Training', 'Testing'} )}
                args.FeatureType        char ...
                        {mustBeMember( args.FeatureType, ...
                                   {'Gaussian'} )} = 'Gaussian'
                args.ClassSizes         double ...
                        {mustBeInteger, mustBePositive} = 500
                args.ClassElements      double ...
                        {mustBeInteger, mustBePositive} = 2
                args.ClassMeans         double
                args.ClassSDs           double
                args.ClassPeaks         double
                args.MeanCovariance     cell
                args.SDCovariance       cell
                args.PeakCovariance     cell
                args.Noise              double = 0.002
                args.HasVariableLength  logical = false
                args.TerminationValue   double = 0
                args.PaddingLength      double = 0
                superArgs.?modelDataset
            end

            args = validateArgs( args );

            % initialize the chosen feature function
            switch args.FeatureType
                case 'Gaussian'
                    featureFcn = @(x, ampl, mu, sigma) ...
                        (ampl/sqrt((2*pi))*exp(-0.5*((x-mu)/sigma).^2));
            end

            % setup the timespan
            nPts = 501;
            args.tSpan = linspace( -5, 5, nPts )';

            % initialize the data arrays
            nObs = sum(args.ClassSizes);
            X = cell( nObs, 1 );
            Y = zeros( nObs, 1 );

            % iterate through the classes, generating data
            idxStart = 1;
            for i = 1:length( args.ClassSizes )

                idxEnd = idxStart + args.ClassSizes(i) - 1;

                X( idxStart:idxEnd ) = generateData( i, featureFcn, args );
                Y( idxStart:idxEnd ) = i;

                idxStart = idxEnd + 1;

            end

            % setup padding
            if args.PaddingLength==0
                pad.length = length( args.tSpan );
            else
                pad.length = args.PaddingLength;
            end
            pad.longest = false;
            pad.location = 'Left';
            pad.value = 1;
            pad.same = true;
            pad.anchoring = 'None';

            paramsFd.tSpan= args.tSpan;
        
            % setup fda
            paramsFd.basisOrder = 4;
            paramsFd.penaltyOrder = 2;
            paramsFd.lambda = 1E-2;
         
            % process the data and complete the initialization
            superArgsCell = namedargs2cell( superArgs );

            name = 'Exemplar Data';

            self = self@modelDataset( X, Y, superArgsCell{:}, ...
                            padding = pad, ...
                            fda = paramsFd, ...
                            datasetName = name, ...
                            channelLabels = "Y (no units)", ...
                            timeLabel = "Time (no units)", ...
                            channelLimits = [0 2] );

            self.FeatureType = args.FeatureType;
            self.ClassSizes = args.ClassSizes;
            self.ClassElements = args.ClassElements;
            self.ClassMeans = args.ClassMeans;
            self.ClassSDs = args.ClassSDs;
            self.ClassPeaks = args.ClassPeaks;
            self.MeanCovariance = args.MeanCovariance;
            self.SDCovariance = args.SDCovariance;
            self.PeakCovariance = args.PeakCovariance;
            self.Noise = args.Noise;
            self.HasVariableLength = args.HasVariableLength;
            self.TerminationValue = args.TerminationValue;

        end

    end

end


function [ X, Y ] = generateData( c, fcn, args )
    % Generate gaussian shaped curves
    arguments
        c             double
        fcn             function_handle
        args            struct
    end

    % initialise the array holding the generated data
    nPts = length( args.tSpan );

    nObs = sum( args.ClassSizes(c) );
    X = cell( nObs, 1 );
    for i = 1:nObs
        X{i} = zeros( nPts, 1 );
    end

    % generate the parameters according to mean and covariance
    ampl  = mvnrnd( args.ClassPeaks( c, : ), ...
                    args.PeakCovariance{c}, ...
                    nObs );
    mu    = mvnrnd( args.ClassMeans( c, : ), ...
                    args.MeanCovariance{c}, ...
                    nObs );
    sigma = mvnrnd( args.ClassSDs( c, : ), ...
                    args.SDCovariance{c}, ...
                    nObs );
    % produce the curves by layering on feature elements
    for j = 1:args.ClassElements(c)

        for i = 1:nObs

            X{i} = X{i} + fcn( args.tSpan, ampl(i,j), mu(i,j), sigma(i,j) );

        end

    end

    % truncate the series, if required
    if args.HasVariableLength
        for i = 1:nObs
            adjX = X{i}-args.TerminationValue;
            prod = adjX(1:end-1).*adjX(2:end);
            first = find( prod < 0, 1 );
            if isempty( first )
                first = 1;
            end
            last = find( prod < 0, 1, 'last' );
            if isempty( last ) || last==first
                last = length(X{i});
            end
            X{i} = X{i}( first:last );
        end

    end

    % add in noise
    for i = 1:nObs
        X{i} = X{i} + args.Noise*randn( size(X{i}) );
    end

end




function args = validateArgs( args )

    nClasses = length( args.ClassSizes );
    nElements = args.ClassElements;
    
    if ~isfield( args, 'ClassMeans' )
        args.ClassMeans = repmat(linspace( -2, 2, nElements ), nClasses, 1);
    end
    
    if ~isfield( args, 'ClassSDs' )
        args.ClassSDs = ones( nClasses, nElements );
    end

    if ~isfield( args, 'ClassPeaks' )
        args.ClassPeaks = 2*ones( nClasses, nElements );
    end

    if ~isfield( args, 'MeanCovariance' )
        args.MeanCovariance = setCovariance( 0.1, nClasses, nElements  );
    end

    if ~isfield( args, 'SDCovariance' )
        args.SDCovariance = setCovariance( 0.1, nClasses, nElements  );
    end

    if ~isfield( args, 'PeakCovariance' )
        args.PeakCovariance = setCovariance( 0.25, nClasses, nElements  );
    end

    invalid =    size( args.ClassMeans, 1 ) ~= nClasses ...
              || size( args.ClassSDs, 1 ) ~= nClasses ...
              || size( args.ClassPeaks, 1 ) ~= nClasses ...
              || length( args.MeanCovariance ) ~= nClasses ...
              || length( args.SDCovariance ) ~= nClasses ...
              || length( args.PeakCovariance ) ~= nClasses;
    
    if invalid
        eid = 'exemplarDataset:InvalidParameters';
        msg = 'The parameters do not all have the same number of classes.';
        throwAsCaller( MException(eid,msg) );
    end

    invalid =    size( args.ClassMeans, 2 ) ~= nElements ...
              || size( args.ClassSDs, 2 ) ~= nElements ...
              || size( args.ClassPeaks, 2 ) ~= nElements ...
              || length( args.MeanCovariance{1} ) ~= nElements ...
              || length( args.SDCovariance{1} ) ~= nElements ...
              || length( args.PeakCovariance{1} ) ~= nElements;
    
    if invalid
        eid = 'exemplarDataset:InvalidParameters';
        msg = 'The parameters do not all have the same number of elements.';
        throwAsCaller( MException(eid,msg) );
    end

    for i = 1:nClasses

        checkCovariance( args.MeanCovariance{i}, 'Mean' );
        checkCovariance( args.SDCovariance{i}, 'SD' );
        checkCovariance( args.PeakCovariance{i}, 'Peak' );

    end

end


function C = setCovariance( a, c, e )

    C = cell( c, 1  );
    for i = 1:c
        C{i} = a*eye( e );
    end

end


function checkCovariance( C, name )

    if ~issymmetric( C )
        eid = 'exemplarDataset:NonSymmetric';
        msg = [ 'The ' name ' covariance matrix is not symmetric.' ];
        throwAsCaller( MException(eid,msg) );
    end

    try
        chol( C );
    catch
        eid = 'exemplarDataset:NotPositiveDefinite';
        msg = [ 'The ' name ' covariance matrix is not positive definite.' ];
        throwAsCaller( MException(eid,msg) );
    end

end