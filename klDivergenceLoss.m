% ************************************************************************
% Class: klDivergenceLoss
%
% Subclass for Kullback-Leibler divergence loss
%
% ************************************************************************

classdef klDivergenceLoss < lossFunction

    properties

    end

    methods

        function self = klDivergenceLoss( name, superArgs )
            % Initialize the loss function
            arguments
                name                 char {mustBeText}
                superArgs.?lossFunction
            end

            superArgsCell = namedargs2cell( superArgs );
            self = self@lossFunction( name, superArgsCell{:}, ...
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