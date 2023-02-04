classdef ReconstructionLoss < LossFunction
    % Subclass for reconstruction error
    properties
        Scale
    end

    methods

        function self = ReconstructionLoss( name, superArgs, args )
            % Initialize the loss function
            arguments
                name                 char {mustBeText}
                superArgs.?LossFunction
                args.input           char = 'X-XHat'
                args.yLim            double = [0, 0.25]
            end

            superArgsCell = namedargs2cell( superArgs );
            self = self@LossFunction( name, superArgsCell{:}, ...
                                 type = 'Reconstruction', ...
                                 input = args.input, ...
                                 lossNets = {'Encoder', 'Decoder','ZClassifier'}, ...
                                 yLim = args.yLim );

            self.Scale = 1;

        end


        function self = setScale( self, data )
            % Set the scaling factor when calculating reconstruction loss
            arguments
                self        ReconstructionLoss
                data        double
            end

            scaling = squeeze(mean(var( data )))';
            self.Scale = scaling;

        end


        function loss = calcLoss( self, X, XHat )
            % Calculate the reconstruction loss
            arguments
                self        ReconstructionLoss
                X           
                XHat
            end

            loss = reconLoss( X, XHat, self.Scale );
    
        end


        function loss = calcTemporalLoss( self, X, XHat )
            % Compute the mean squared error as a function of time
            arguments
                self        ReconstructionLoss
                X           
                XHat
            end

            loss = reconTemporalLoss( X, XHat, self.Scale );

        end

    end

end