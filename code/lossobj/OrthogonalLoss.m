classdef OrthogonalLoss < LossFunction
    % Subclass for the Z othogonality loss (penalising correlated latent codes)
    % Code adapted from https://github.com/WangDavey/COAE

    properties

    end

    methods

        function self = OrthogonalLoss( name, superArgs )
            % Initialize the loss function
            arguments
                name                 char {mustBeText}
                superArgs.?LossFunction
            end

            superArgsCell = namedargs2cell( superArgs );
            self = self@LossFunction( name, superArgsCell{:}, ...
                                 type = 'Regularization', ...
                                 input = 'Z', ...
                                 lossNets = {'Encoder'} );

        end


        function loss = calcLoss( self, dlZ )
            % Calculate the orthogonality loss
            arguments
                self        OrthogonalLoss
                dlZ         dlarray
            end
            
            % get the variance and covariance
            [ dlVar, dlCoVar ] = dlVarianceCovariance( dlZ );

            % penalise high covariance
            loss = mean( dlCoVar.^2, 'all' );
            % and variation in variance between components
            % loss = loss + var( dlVar );

        end


    end


end
