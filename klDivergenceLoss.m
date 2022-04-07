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
                                 input = 'Z' );

        end

    end

    methods (Static)

        function loss = calcLoss( self, Z )
            % Calculate the KL divergence
            if self.doCalcLoss
    
                ZSigma = std( Z );
                ZMu = mean( Z );
                
                loss = 0.5*sum( ZSigma.^2 + ZMu.^2 - 1 - log(ZSigma.^2) );
    
            else
                loss = 0;
            end
    
        end

    end

end