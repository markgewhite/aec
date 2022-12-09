classdef SyntheticDataset < ModelDataset
    % Subclass for generating a simulated (artificial) dataset
    % based originally on the method proposed by Hsieh et al. (2021).
    % Enhanced with option to have multiple basis levels.
    % The number of levels is specified if basis is a cell array

    properties
        TemplateSeed    % seed specifying the data set template
        DatasetSeed     % seed specifying the random realization of that template
        ClassSizes      % number observations per class (vector)
        NumPts          % number of point across domain
        Scaling         % scaling ratio of the levels
        Mu              % mean amplitudes across levels
        Sigma           % standard deviation in magnitudes
        Eta             % noise 
        Tau             % the degree of time warping
        SharedLevel     % the level at which ...
        WarpLevel       % the level at which time warping is applied
    end

    methods

        function self = SyntheticDataset( set, args, superArgs )
            % Generate the synthetic dataset
            arguments
                set                 char ...
                    {mustBeMember( set, ...
                                   {'Training', 'Testing'} )}
                args.TemplateSeed    double = 1234
                args.DatasetSeed     double = 9876
                args.ClassSizes     double ...
                    {mustBeInteger, mustBePositive} = [500 500 500]
                args.NumPts         double = 101
                args.NumTemplatePts double ...
                    {mustBeInteger, mustBePositive} = 17
                args.Channels       double = 1
                args.Scaling        double = [2 4 8]
                args.Mu             double = [1 2 4]
                args.Sigma          double = [1 1 1]
                args.Eta            double = 0.1
                args.Tau            double = 0
                args.SharedLevel    double = 3
                args.WarpLevel      double = 2
                args.PaddingLength  double = 0
                args.Lambda         double = []
                superArgs.?ModelDataset
            end

            % setup the timespan
            args.TSpanTemplate = linspace( 0, 1, args.NumTemplatePts )';
            args.TSpan = linspace( 0, 1, args.NumPts )';

            % set the random seed
            switch set
                case 'Training'
                    args.RandomSeed = args.TemplateSeed + args.DatasetSeed;
                case 'Testing'
                    args.RandomSeed = args.TemplateSeed + args.DatasetSeed + 1;
            end

            [ XRaw, Y ] = generateData( args.ClassSizes, args );

            % setup padding
            if args.PaddingLength==0
                pad.Length = length( args.TSpan );
            else
                pad.Length = args.PaddingLength;
            end

            pad.Longest = false;
            pad.Location = 'Left';
            pad.Value = 1;
            pad.Same = true;
            pad.Anchoring = 'None';
         
            % process the data and complete the initialization
            superArgsCell = namedargs2cell( superArgs );
            nClasses = length( args.ClassSizes );
            labels = strings( nClasses, 1 );
            for i = 1:nClasses
                labels(i) = strcat( "Class ", char(64+i) );
            end

            self = self@ModelDataset( XRaw, Y, args.TSpan, ...
                            superArgsCell{:}, ...
                            padding = pad, ...
                            lambda = args.Lambda, ...
                            datasetName = "Synthetic Data", ...
                            channelLabels = "X (no units)", ...
                            timeLabel = "Time Domain", ...
                            classLabels = labels, ...
                            channelLimits = [-3 3] );

            self.ClassSizes = args.ClassSizes;
            self.Scaling = args.Scaling;
            self.Mu = args.Mu;
            self.Sigma = args.Sigma;
            self.Eta = args.Eta;
            self.Tau = args.Tau;
            self.SharedLevel = args.SharedLevel;
            self.WarpLevel = args.WarpLevel;
            

        end

    end

end


function [ X, Y ] = generateData( nObs, args )
    % Generate the synthetic data
    arguments
        nObs            double ...
                        {mustBeInteger, mustBePositive}
        args            struct
    end

    % initialise the number of points across multiple layers
    % allow extra space either end for extrapolation when time warping
    % (the time domains are twice as long)
    nLevels = length( args.Scaling );
    
    nPts = zeros( nLevels, 1 );
    tSpanLevels = cell( nLevels, 1 );
    range = [ args.TSpanTemplate(1), args.TSpanTemplate(end) ];
    extra = 0.5*(range(2)-range(1));
    dt = args.TSpanTemplate(2)-args.TSpanTemplate(1);
    
    for j = 1:nLevels
        nPts(j) = 2*((length( args.TSpanTemplate )-1)/args.Scaling(j))+1;
        tSpanLevels{j} = linspace( range(1)-extra, ...
                                   range(2)+extra, ...
                                   nPts(j) )';
    end
    
    nPtsFull = nPts(end);
    tSpanLevelsFull = tSpanLevels{end};

    % set the warping timespan
    tWarp0 = tSpanLevels{ args.WarpLevel };
    
    % initialise the template array across levels
    template = zeros( nPtsFull, args.Channels, nLevels );
    
    % initialise the array holding the generated data
    X = zeros( length( args.TSpan ), sum(nObs), args.Channels );
    Y = zeros( sum(nObs), 1 );
    
    % define the common template shared by all classes
    rng( args.TemplateSeed );
    for j = 1:args.SharedLevel
        template( :,:,j ) = interpRandSeries( tSpanLevels{j}, ...
                                              tSpanLevelsFull, ...
                                              nPts(j), ...
                                              args.Channels, 2 );
    end
    
    a = 0;
    for c = 1:length(nObs)
    
        % define the class-specific template on top
        
        for j = args.SharedLevel+1:nLevels
            template( :,:,j ) = interpRandSeries( tSpanLevels{j}, ...
                                                  tSpanLevelsFull, ...
                                                  nPts(j), ...
                                                  args.Channels, 2 );
        end

        % preserve the template random state
        rngStateTemplate = rng;
        % and switch to the dataset random stream
        if c==1
            rng( args.RandomSeed );
        else
            rng( rngStateData );
        end

        for i = 1:nObs(c)
    
            a = a+1;
    
            % vary the template function across levels
            sample = zeros( nPts(end), args.Channels );
            for j = 1:nLevels 
                sample = sample + ...
                    (args.Mu(j) + randn*args.Sigma(j))*template( :,:,j );
            end
    
            % introduce noise
            sample = sample + args.Eta*randn( nPtsFull, args.Channels );
    
            % warp the time domain at the top level, ensuring monotonicity
            % and avoiding excessive curvature by constraining the gradient
            monotonic = false;
            excessCurvature = false;
            while ~monotonic || excessCurvature

                % generate a time warp series based at the desired level 
                tWarp = tWarp0 + args.Tau*randSeries( 1, length(tWarp0) )';
                
                % interpolate to the most detailed level
                tWarp = interp1( tWarp0, tWarp, tSpanLevelsFull, 'spline' );

                % check constraints
                grad = diff( tWarp )/dt;
                monotonic = all( grad>0 );
                excessCurvature = any( grad<0.2 );

            end
    
            % interpolate using the warped time points
            % as if the points regularly spaced
            sample = interp1( tWarp, sample, tSpanLevelsFull, 'spline' );

            % interpolate to obtain time series of overall required length
            X( :, a, : ) = interp1( tSpanLevelsFull, sample, ...
                                    args.TSpan, 'spline' );

            % add the class information
            Y( a ) = c;
                   
        end

        % preserve the dataset random state
        rngStateData = rng;
        % restore the template random stream
        rng( rngStateTemplate );
    
    end 

    % convert to cell array
    X = num2cell( X, 1 );
    
end




