classdef exemplarDataset < modelDataset
    % Subclass for generating an exemplar dataset
    % for testing the capabilities of different models

    properties
        ClassSizes          % number observations per class (vector)
        ClassMeans          % positions of classes
        ClassSDs            % standard deviations of classes
        ClassPeaks          % amplitudes of classes
        MeanVariations      % within-class position variations
        SDVariations        % within-class SD variations
        PeakVariations      % within-class peak variations
        Noise               % noise 
    end

    methods

        function self = exemplarDataset( set, args, superArgs )
            % Initialize the exemplar dataset
            arguments
                set                 char ...
                    {mustBeMember( set, ...
                                   {'Training', 'Testing'} )}
                args.ClassSizes     double ...
                    {mustBeInteger, mustBePositive} = 500
                args.ClassMeans     double
                args.ClassSDs       double
                args.ClassPeaks     double
                args.MeanVariations double
                args.SDVariations   double
                args.PeakVariations double
                args.Noise          double = 0.002
                superArgs.?modelDataset
            end

            args = validateArgs( args );

            nPts = 501;
            args.tSpan = linspace( -5, 5, nPts );

            [ X, Y ] = generateData( args );

            % convert to cell array
            X = num2cell( X, 1 );

            % setup padding
            pad.length = length( args.tSpan );
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

            self.ClassSizes = args.ClassSizes;
            self.ClassMeans = args.ClassMeans;
            self.ClassSDs = args.ClassSDs;
            self.ClassPeaks = args.ClassPeaks;
            self.MeanVariations = args.MeanVariations;
            self.SDVariations = args.SDVariations;
            self.PeakVariations = args.PeakVariations;
            self.Noise = args.Noise;

        end

    end

end


function args = validateArgs( args )

    nClasses = length( args.ClassSizes );

    if ~isfield( args, 'ClassMeans' )
        args.ClassMeans = linspace( -2, 2, nClasses );
    end
    
    if ~isfield( args, 'ClassSDs' )
        args.ClassSDs = ones( 1, nClasses );
    end

    if ~isfield( args, 'ClassPeaks' )
        args.ClassPeaks = 2*ones( 1, nClasses );
    end

    if ~isfield( args, 'MeanVariations' )
        args.MeanVariations = 0.1*ones( 1, nClasses );
    end

    if ~isfield( args, 'SDVariations' )
        args.SDVariations = 0.1*ones( 1, nClasses );
    end

    if ~isfield( args, 'PeakVariations' )
        args.PeakVariations = 0.25*ones( 1, nClasses );
    end

    invalid =    length( args.ClassMeans ) ~= nClasses ...
              || length( args.ClassSDs ) ~= nClasses ...
              || length( args.ClassPeaks ) ~= nClasses ...
              || length( args.MeanVariations ) ~= nClasses ...
              || length( args.SDVariations ) ~= nClasses ...
              || length( args.PeakVariations ) ~= nClasses;

    if invalid
        eid = 'exemplarDataset:InvalidParameters';
        msg = 'The parameters do not all have the same number of elements.';
        throwAsCaller( MException(eid,msg) );
    end

end

function [ X, Y ] = generateData( args )
    % Generate gaussian shaped curves
    arguments
        args            struct
    end

    % initialize the function
    normFcn = @(x, ampl, mu, sigma) ...
                (ampl/sqrt((2*pi))*exp(-0.5*((x-mu)/sigma).^2));

    % initialise the array holding the generated data
    nPts = length( args.tSpan );
    nObs = sum( args.ClassSizes );
    X = zeros( nPts, nObs );
    Y = zeros( sum(nObs), 1 );

    k = 0;
    for i = 1:length( args.ClassSizes )

        % set the class amplitude, mean and standard deviation
        ampl0 = args.ClassPeaks(i);
        mu0 = args.ClassMeans(i);
        sigma0 = args.ClassSDs(i);

        for j = 1:args.ClassSizes(i)

            k = k+1;
            ampl = abs( ampl0 + args.PeakVariations(i)*randn );
            mu = mu0 + args.MeanVariations(i)*randn;
            sigma = abs( sigma0 + args.SDVariations(i)*randn );

            X( :, k ) = normFcn( args.tSpan, ampl, mu, sigma );

            X( :, k ) = X( : , k ) + args.Noise*randn(nPts,1);

        end

        Y( k-args.ClassSizes(i)+1:k ) = i;

    end

end



