classdef KLDivergenceLoss < LossFunction
    % Subclass for Kullback-Leibler divergence loss

    properties

    end

    methods

        function self = KLDivergenceLoss( name, superArgs )
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

    end

    methods (Static)

        function loss = calcLoss( dlZ )
            % Calculate the KL divergence
            arguments
                dlZ   dlarray
            end

            dlZSigma = std( dlZ );
            dlZMu = mean( dlZ );
            
            loss = 0.5*sum( dlZSigma.^2 + dlZMu.^2 - 1 - log(dlZSigma.^2) );

        end

    end

end