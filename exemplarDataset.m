classdef exemplarDataset < modelDataset
    % Subclass for generating an exemplar dataset
    % for testing the capabilities of different models

    properties
        ClassSizes          % number observations per class (vector)
        BetweenClassScale   % parameters' SD between-classes variation
        WithinClassScale    % parameters' SD between-classes variation
        Noise               % noise 
    end

    methods

        function self = exemplarDataset( set, args, superArgs )
            % Initialize the exemplar dataset
            arguments
                set                 char ...
                    {mustBeMember( set, ...
                                   {'Training', 'Testing'} )}
                args.Type           char ...
                    {mustBeMember( args.Type, ...
                       {'Gaussian'} )} = 'Gaussian'
                args.ClassSizes     double ...
                    {mustBeInteger, mustBePositive} = [500 500 500]
                args.BetweenClassScale   double = 0.5
                args.WithinClassScale    double = 0.25
                args.Noise          double = 0.002
                superArgs.?modelDataset
            end

            nPts = 501;
            args.tSpan = linspace( -5, 5, nPts );

            switch args.Type
                case 'Gaussian'
                    [ X, Y ] = ...
                        exemplarDataset.gaussianData( args );

            end

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

            name = ['Exemplar Data (' args.Type ')'];

            self = self@modelDataset( X, Y, superArgsCell{:}, ...
                            padding = pad, ...
                            fda = paramsFd, ...
                            datasetName = name, ...
                            channelLabels = "Y (no units)", ...
                            timeLabel = "Time (no units)", ...
                            channelLimits = [0 2.5] );

            self.ClassSizes = args.ClassSizes;
            self.BetweenClassScale = args.BetweenClassScale;
            self.WithinClassScale = args.WithinClassScale;
            self.Noise = args.Noise;
            

        end

    end


    methods (Static)

        function [ X, Y ] = gaussianData( args )
            % Generate gaussian shaped curves
            arguments
                args            struct
            end
    
            % initialize the function
            normFcn = @(x, ampl, mu, sigma) ...
                        (ampl/sqrt((2*pi))*exp(-0.5*((x-mu)/sigma).^2));

            % for convenience use short meaningful coefficients
            noise = args.Noise;
            btwScale = args.BetweenClassScale;
            withinScale = args.WithinClassScale;

            % initialise the array holding the generated data
            nPts = length( args.tSpan );
            nObs = sum( args.ClassSizes );
            X = zeros( nPts, nObs );
            Y = zeros( sum(nObs), 1 );

            k = 0;
            for c = 1:length( args.ClassSizes )

                % set the class amplitude, mean and standard deviation
                ampl0 = 2*(2+ btwScale*randn);
                mu0 = randn;
                sigma0 = 1 + 0.1*btwScale*randn;

                for i = 1:args.ClassSizes(c)

                    k = k+1;
                    ampl = abs( ampl0 + withinScale*randn );
                    mu = mu0 + withinScale*randn;
                    sigma = abs( sigma0 + withinScale*randn );

                    X( :, k ) = normFcn( args.tSpan, ampl, mu, sigma );

                    X( :, k ) = X( : , k ) + noise*randn(nPts,1);

                end

                Y( k-args.ClassSizes(c)+1:k ) = c;

            end

        end



   end


end


