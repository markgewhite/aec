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
            
            % convert to double for speed
            % (other loss functions will ensure tracing)
            Z = double(extractdata( dlZ ));

            % use Pearson's product moment correlation
            R = corr( Z' );

            % calculate the loss by removing the diagonal
            d = length(R);
            loss = sum( ( R-eye(d) ).^2, 'all' )/(d*(d-1));

        end


    end


end
