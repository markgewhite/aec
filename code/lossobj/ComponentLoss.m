classdef ComponentLoss < LossFunction
    % Subclass for the loss functions computed from AE components

    properties
        Criterion     % criterion function for component loss
        Sampling      % sampling method for Z offsets
        NumSamples    % number of samples to draw to generate components
        MaxObservations % limit to batch size when training
        Alpha         % hyperparameter regulating magnitude
        Scale         % scaling factor between channels
    end

    methods

        function self = ComponentLoss( name, args, superArgs )
            % Initialize the loss function
            arguments
                name                 char {mustBeText}
                args.Criterion       char ...
                    {mustBeMember( args.Criterion, ...
                        {'InnerProduct', 'Orthogonality', ...
                         'Varimax', 'ExplainedVariance'} )} ...
                                        = 'InnerProduct'
                args.Sampling        char ...
                    {mustBeMember( args.Sampling, ...
                        {'Regular', 'Component'} )} = 'Component'
                args.NumSamples      double ...
                    {mustBeInteger, mustBePositive} = 5
                args.MaxObservations double ...
                    {mustBeInteger, mustBePositive} = 500
                args.Alpha           double = 1E2
                args.YLim            double = [-0.10, 0.02]
                superArgs.?LossFunction
            end

            superArgsCell = namedargs2cell( superArgs );
            self = self@LossFunction( name, superArgsCell{:}, ...
                                 type = 'Component', ...
                                 input = 'XC', ...
                                 lossNets = {'Encoder', 'Decoder'}, ...
                                 ylim = args.YLim );

            self.Criterion = args.Criterion;
            self.Sampling = args.Sampling;
            self.NumSamples = args.NumSamples;
            self.MaxObservations = args.MaxObservations;
            self.Alpha = args.Alpha;
            self.Scale = 1;

        end


        function self = setScale( self, data )
            % Set the scaling factor when calculating reconstruction loss
            arguments
                self        ComponentLoss
                data        double
            end

            scaling = squeeze(mean(var( data )))';
            self.Scale = scaling;

        end


        function loss = calcLoss( self, dlXC )
            % Calculate the component loss
            arguments
                self
                dlXC  dlarray  % generated AE components
            end
               
            switch self.Criterion
                case 'InnerProduct'
                    % compute the inner product as a test
                    % of component orthogonality
                    loss = self.Alpha*innerProductLoss( dlXC );

                case 'Orthogonality'
                    % compute the inner product as a test
                    % of component orthogonality
                    loss = self.Alpha*orthogonality( dlXC );

                case 'Varimax'
                    % compute the component variance across 
                    % its length, penalising low variance
                    loss = varimax( dlXC );

                case 'ExplainedVariance'
                    % compute the component variance across 
                    % its length, penalising high variance
                    loss = explainedVariance( dlXC, nChannels, nComp, ...
                                         nSamples, self.Scale );

            end

        end

    end

end


function loss = orthogonality( dlXC )
    % Calculate a pseudo orthogonality

    [nPts, nSamples, nComp, nChannels] = size( dlXC );

    for c = 1:nChannels
        for k = 1:nSamples
            dlXCsample = squeeze( dlXC(:,k,:,c) );
            %dlXCsample = permute( dlXCsample, [2 1] );
            [dlVar, dlCov] = dlVarianceCovariance( dlXCsample );
            dlCorr = dlCorrelation( dlXCsample );
            if k==1 && c==1
                orth = sum( dlCorr.^2, 'all' ) ...
                            + 0*var(abs(dlVar)) - (nPts-1)*sum(dlVar.^2);
            else
                orth = orth + sum( dlCorr.^2, 'all' ) ...
                            + 0*var(abs(dlVar)) - (nPts-1)*sum(dlVar.^2);
            end
        end
    end

    loss = orth/(nChannels*nSamples*nPts*(nPts-1));

end


function loss = varimax( XC )
    % Calculate the varimax loss which is the
    % negative mean of component variances

    nObs = size( XC, 2 );
    nChannels = size( XC, 3 );
    var = zeros( nChannels, 1 );
    for i = 1:nObs
        var = var + std( XC(:,i), [], 1 ).^2;
    end

    var = var./scale;

    loss = -0.01*mean(var)/nObs;   

end


function loss = explainedVariance( XC, nChannels, nComps, nSamples, scale )
    % Calculate the explained variance loss 

    var = zeros( 1, nChannels );

    for c = 1:nChannels
        for k = 1:nSamples
            for i = 1:nComps
                var(c) = var(c) + mean( XC(:,c,k,i).^2 );
            end
        end
    end

    var = var./scale;

    loss = 1 - mean(var)/(nChannels*nComps*nSamples);


end

