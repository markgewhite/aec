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
                        {'Orthogonality', 'Varimax'} )} ...
                                        = 'Orthogonality'
                args.Sampling        char ...
                    {mustBeMember( args.Sampling, ...
                        {'Regular', 'Component'} )} = 'Component'
                args.NumSamples      double ...
                    {mustBeInteger, mustBePositive} = 5
                args.MaxObservations double ...
                    {mustBeInteger, mustBePositive} = 500
                args.Alpha           double = 1E0
                args.YLim            double = [0, 0.2]
                superArgs.?LossFunction
            end

            superArgsCell = namedargs2cell( superArgs );
            self = self@LossFunction( name, superArgsCell{:}, ...
                                 type = 'Component', ...
                                 input = {'dlXC'}, ...
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
                case 'Orthogonality'
                    % compute the inner product as a test
                    % of component orthogonality
                    loss = self.Alpha*orthogonality( dlXC );

                case 'Varimax'
                    % compute the component variance across 
                    % its length, penalising low variance
                    loss = self.Alpha*varimax( dlXC );

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
            dlXCsample = permute( dlXCsample, [2 1] );
            dlCorr = dlCorrelation( dlXCsample );
            if k==1 && c==1
                orth = mean( dlCorr.^2, 'all' );
            else
                orth = orth + mean( dlCorr.^2, 'all' );
            end
        end
    end

    loss = orth/(nChannels*nSamples*nComp*(nComp-1));

end


function loss = varimax( dlXC )
    % Calculate the varimax loss which is the 
    % mean square of the component variances

    loss = mean( dlXC.^2, 'all' );

end


