classdef OrthogonalLoss < LossFunction
    % Subclass for the Z othogonality loss (penalising correlated latent codes)
    % Code adapted from https://github.com/WangDavey/COAE

    properties
        Alpha           % scaling factor for loss
    end

    methods

        function self = OrthogonalLoss( name, args, superArgs )
            % Initialize the loss function
            arguments
                name                char {mustBeText}
                args.Alpha          double ...
                    {mustBeGreaterThan(args.Alpha,0)} = 0.001
                superArgs.?LossFunction
            end

            superArgsCell = namedargs2cell( superArgs );
            self = self@LossFunction( name, superArgsCell{:}, ...
                                 type = 'Regularization', ...
                                 input = 'Z', ...
                                 lossNets = {'Encoder'}, ...
                                 yLim = [0, 0.25]);

            self.Alpha = args.Alpha;

        end


        function loss = calcLoss( self, dlZ )
            % Calculate the orthogonality loss
            arguments
                self        OrthogonalLoss
                dlZ         dlarray
            end
            
            % get the variance and covariance
            [ ~, dlCoVar ] = dlVarianceCovariance( dlZ );

            % penalise high covariance
            loss = self.Alpha*mean( dlCoVar.^2, 'all' );


        end


    end


end
