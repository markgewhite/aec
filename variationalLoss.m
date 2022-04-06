% ************************************************************************
% Class: variationalLoss
%
% Subclass for Kullback-Leibler divergence loss 
% for variational autoencoders where the mean and variance are 
% obtaine directly from the encoder network
%
% ************************************************************************

classdef variationalLoss < lossFcn

    properties

    end

    methods

        function self = variationalLoss( name, superArgs )
            % Initialize the loss function
            arguments
                name                 char {mustBeText}
                superArgs.?lossFcn
            end

            superArgsCell = namedargs2cell( superArgs );
            self = self@lossFcn( name, superArgsCell{:}, ...
                                 type = 'Regularization' );

        end

    end

    methods (Static)

        function loss = calcLoss( self, ZMu, ZLogVar )
            % Calculate the KL divergence
            if self.doCalcLoss
                
                loss = 0.5*sum( exp(ZLogVar) + ZMu.^2 - 1 - ZLogVar );
    
            else
                loss = 0;
            end

    end

    end

end