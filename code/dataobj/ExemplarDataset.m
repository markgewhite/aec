classdef ExemplarDataset < ModelDataset
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
        TerminationValues   % truncate when passing through value [lower, upper], if variable length
        TerminationType     % truncate when going above or below TerminationValues [lower, upper]
    end

    methods

        function self = ExemplarDataset( set, args, superArgs )
            % Initialize the exemplar dataset
            arguments
                set                     char ...
                        {mustBeMember( set, ...
                                   {'Training', 'Testing'} )}
                args.randomSeeds        (1,2) double = [ 1234 5678 ]
                args.FeatureType        char ...
                        {mustBeMember( args.FeatureType, ...
                               {'Gaussian', 'Sigmoid'} )} = 'Gaussian'
                args.ClassSizes         double ...
                        {mustBeInteger, mustBePositive} = 500
                args.ClassElements      double ...
                        {mustBeInteger, mustBePositive} = 1
                args.ClassMeans         double
                args.ClassSDs           double
                args.ClassPeaks         double
                args.MeanCovariance     cell
                args.SDCovariance       cell
                args.PeakCovariance     cell
                args.Noise              double = 0.002
                args.HasVariableLength  logical = false
                args.TerminationValues  (1, 2) double = [0 0]
                args.TerminationTypes   (1, 2) string ...
                        {mustBeMember( args.TerminationTypes, ...
                               {'Above', 'Below'} )} = 'Below'
                args.PaddingLength      double = 0
                args.Lambda             double = []
                superArgs.?ModelDataset
            end

            args = validateArgs( args );

            % initialize the chosen feature function
            switch args.FeatureType
                case 'Gaussian'
                    featureFcn = @(x, ampl, mu, sigma) ...
                                    ampl*exp(-0.5*((x-mu)/sigma).^2);
                case 'Sigmoid'
                    featureFcn = @(x, ampl, mu, sigma) ...
                                    ampl./(exp(-(x-mu)/sigma)+1);
            end

            % setup the timespan
            nPts = 101;
            args.tSpan = linspace( -4, 4, nPts )';

            % initialize the data arrays
            nClasses = length(args.ClassSizes);
            nObs = sum(args.ClassSizes);
            X = cell( nObs, 1 );
            Y = zeros( nObs, 1 );

            % set the random seed
            switch set
                case 'Training'
                    rng( args.randomSeeds(1) );
                case 'Testing'
                    rng( args.randomSeeds(2) );
            end

            % iterate through the classes, generating data
            idxStart = 1;
            labels = strings( nClasses, 1 );
            for i = 1:nClasses

                idxEnd = idxStart + args.ClassSizes(i) - 1;

                X( idxStart:idxEnd ) = generateData( i, featureFcn, args );
                Y( idxStart:idxEnd ) = i;

                idxStart = idxEnd + 1;

                labels(i) = strcat( "Class ", char(64+i) );

            end

            % setup padding
            if args.PaddingLength==0
                pad.Length = length( args.tSpan );
            else
                pad.Length = args.PaddingLength;
            end
            pad.Longest = false;
            pad.Location = 'Left';
            pad.Value = 1;
            pad.Same = true;
            pad.Anchoring = 'None';

            tSpan= args.tSpan;
         
            % process the data and complete the initialization
            superArgsCell = namedargs2cell( superArgs );

            name = "Exemplar Data";

            self = self@ModelDataset( X, Y, tSpan, ...
                            superArgsCell{:}, ...
                            padding = pad, ...
                            lambda = args.Lambda, ...
                            datasetName = name, ...
                            channelLabels = "\bf{\it{x_n(t)}}", ...
                            timeLabel = "\bf{\it{t}}", ...
                            classLabels = labels, ...
                            channelLimits = [-1 5] );

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
            self.TerminationValues = args.TerminationValues;
            self.TerminationTypes = args.TerminationTypes;

            figure(4);
            plot( self.TSpan.Target, self.XTarget );

        end

    end

end


function [ X, Y ] = generateData( c, fcn, args )
    % Generate gaussian shaped curves
    arguments
        c               double
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
    ampl  = mvnrndChi2( args.ClassPeaks( c, : ), ...
                        args.PeakCovariance{c}, ...
                        nObs, 4 );

    mu    = mvnrnd( args.ClassMeans( c, : ), ...
                    args.MeanCovariance{c}, ...
                    nObs );

    sigma = mvnrndChi2( args.ClassSDs( c, : ), ...
                        args.SDCovariance{c}, ...
                        nObs, 4 );  

    % produce the curves by layering on feature elements
    for j = 1:args.ClassElements

        for i = 1:nObs

            X{i} = X{i} + fcn( args.tSpan, ampl(i,j), mu(i,j), sigma(i,j) );

        end

    end

    % add in noise
    for i = 1:nObs
        X{i} = X{i} + args.Noise*randn( size(X{i}) );
    end

    % truncate the series, if required
    if args.HasVariableLength
        for i = 1:nObs
            switch args.TerminationTypes(1)
                case 'Above'
                    first = find( X{i} < args.TerminationValues(1), 1 );
                case 'Below'
                    first = find( X{i} > args.TerminationValues(1), 1 );
            end
            if isempty( first )
                first = 1;
            end
            switch args.TerminationTypes(2)
                case 'Above'
                    last = find( X{i} > args.TerminationValues(2), 1, 'last' );
                case 'Below'
                    last = find( X{i} < args.TerminationValues(2), 1, 'last' );
            end
            if isempty( last ) || last==first
                last = length(X{i});
            end
            X{i} = X{i}( first:last );
        end

    end

end


function y = mvnrndChi2( mu, cov, n, df )
    % Multivariate Chi-squared distribution
    arguments
        mu      double
        cov     double
        n       double {mustBeInteger, mustBePositive}
        df      double {mustBeInteger, mustBePositive} = 4
    end

    y = zeros( n, length(mu), df );
    for k = 1:df
        y(:,:,k) = mvnrnd( mu, cov, n );
    end
    y = sqrt( sum( y.^2, 3 )/df );

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
