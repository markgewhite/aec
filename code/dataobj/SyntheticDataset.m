classdef SyntheticDataset < ModelDataset
    % Subclass for generating a simulated (artificial) dataset
    % based originally on the method proposed by Hsieh et al. (2021).
    % Enhanced with option to have multiple basis levels.
    % The number of levels is specified if basis is a cell array

    properties
        ClassSizes      % number observations per class (vector)
        NumPts          % number of point across domain
        Ratio           % amplitude ratio of different levels
        Mu              % mean amplitudes across levels
        Sigma           % standard deviation in magnitudes
        Eta             % noise 
        Tau             % the degree of time warping
        SharedLevel     % the level at which ...
        WarpLevel       % the level at which time warping is applied
    end

    methods

        function self = SyntheticDataset( set, args, superArgs )
            % Load the countermovement jump GRF dataset
            arguments
                set                 char ...
                    {mustBeMember( set, ...
                                   {'Training', 'Testing'} )}
                args.ClassSizes     double ...
                    {mustBeInteger, mustBePositive} = [500 500 500]
                args.NumPts         double ...
                    {mustBeInteger, mustBePositive} = 17
                args.Channels       double = 1
                args.Ratio          double = [2 4 8]
                args.Mu             double = [1 2 4]
                args.Sigma          double = [1 1 1]
                args.Eta            double = 0.1
                args.Tau            double = 0
                args.SharedLevel    double = 3
                args.WarpLevel      double = 2
                args.PaddingLength  double = 0
                superArgs.?ModelDataset
            end

            args.tSpan = linspace( 0, 1024, args.NumPts );

            [ XRaw, Y ] = SyntheticDataset.genData( args.ClassSizes, args );

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
        
            % setup fda
            paramsFd.BasisOrder = 4;
            paramsFd.PenaltyOrder = 2;
            paramsFd.Lambda = 1E2;
         
            % process the data and complete the initialization
            superArgsCell = namedargs2cell( superArgs );
            nClasses = length( args.ClassSizes );
            labels = strings( nClasses, 1 );
            for i = 1:nClasses
                labels(i) = strcat( "Class ", char(64+i) );
            end

            self = self@ModelDataset( XRaw, Y, tSpan, ...
                            superArgsCell{:}, ...
                            padding = pad, ...
                            fda = paramsFd, ...
                            datasetName = "Synthetic Data", ...
                            channelLabels = "X (no units)", ...
                            timeLabel = "Time Domain", ...
                            classLabels = labels, ...
                            channelLimits = [-5 5] );

            self.ClassSizes = args.ClassSizes;
            self.Ratio = args.Ratio;
            self.Mu = args.Mu;
            self.Sigma = args.Sigma;
            self.Eta = args.Eta;
            self.Tau = args.Tau;
            self.SharedLevel = args.SharedLevel;
            self.WarpLevel = args.WarpLevel;
            

        end

    end


    methods (Static)

        function [ X, Y ] = genData( nObs, args )
            % Generate the data
            arguments
                nObs            double ...
                                {mustBeInteger, mustBePositive}
                args            struct
            end
        
            % initialise the number of points across multiple layers
            % allow extra space either end for extrapolation when time warping
            % (the time domains are twice as long)
            nLevels = length( args.Ratio );
            
            nPts = zeros( nLevels, 1 );
            tSpanLevels = cell( nLevels, 1 );
            range = [ args.tSpan(1), args.tSpan(end) ];
            extra = 0.5*(range(2)-range(1));
            dt = args.tSpan(2)-args.tSpan(1);
            
            for j = 1:nLevels
                nPts(j) = 2*((length( args.tSpan )-1)/args.Ratio(j))+1;
                tSpanLevels{j} = linspace( range(1)-extra, ...
                                           range(2)+extra, ...
                                           nPts(j) )';
            end
            
            tWarp0 = tSpanLevels{ args.WarpLevel };
            
            % initialise the template array across levels
            template = zeros( nPts(1), args.Channels, nLevels );
            
            % initialise the array holding the generated data
            X = zeros( length( args.tSpan ), sum(nObs), args.Channels );
            Y = zeros( sum(nObs), 1 );
            
            % define the common template shared by all classes
            for j = args.SharedLevel:nLevels
                template( :,:,j ) = interpRandSeries( tSpanLevels{j}, ...
                                                      tSpanLevels{1}, ...
                                                      nPts(j), ...
                                                      args.Channels, 2 );
            end
            
            a = 0;
            for c = 1:length(nObs)
            
                % generate random template function coefficients
                % with covariance between the series elements
                % interpolating to the base layer (1)
                for j = 1:args.SharedLevel-1
                    template( :,:,j ) = interpRandSeries( tSpanLevels{j}, ...
                                                          tSpanLevels{1}, ...
                                                          nPts(j), ...
                                                          args.Channels, 2 );
                end
               
                for i = 1:nObs(c)
            
                    a = a+1;
            
                    % vary the template function across levels
                    sample = zeros( nPts(1), args.Channels );
                    for j = 1:nLevels 
                        sample = sample + ...
                            (args.Mu(j) + args.Sigma(j)*randn(1,1)) ...
                                            * template( :,:,j );
                    end
            
                    % introduce noise
                    sample = sample + args.Eta*randn( nPts(1), args.Channels );
            
                    % warp the time domain at the top level, ensuring monotonicity
                    % and avoiding excessive curvature by constraining the gradient
                    monotonic = false;
                    excessCurvature = false;
                    while ~monotonic || excessCurvature

                        % generate a time warp series based on the top-level 
                        tWarp = tWarp0 + args.Tau*randSeries( 1, length(tWarp0) )';
                        
                        % interpolate so it fits the initial level
                        tWarp = interp1( tWarp0, tWarp, tSpanLevels{1}, 'spline' );

                        % check constraints
                        grad = diff( tWarp )/dt;
                        monotonic = all( grad>0 );
                        excessCurvature = any( grad<0.2 );

                    end
            
                    % interpolate the coefficients to the warped time points
                    X( :, a, : ) = interp1( tWarp, sample, args.tSpan, 'spline' );

                    % add the class information
                    Y( a ) = c;
                           
                end
            
            end

            % convert to cell array
            X = num2cell( X, 1 );
            
            end


   end


end


