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
            
            % get the correlation matrix
            dlR = dlCorrelation( dlZ ); 

            loss = mean( ( dlR - eye(size(dlR,1)) ).^2, 'all' );

        end


    end


end
