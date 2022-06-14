classdef VariationalLoss < LossFunction
    % Subclass for Kullback-Leibler divergence loss 
    % for variational autoencoders where the mean and variance are 
    % obtaine directly from the encoder network

    properties

    end

    methods

        function self = VariationalLoss( name, superArgs )
            % Initialize the loss function
            arguments
                name                 char {mustBeText}
                superArgs.?LossFunction
            end

            superArgsCell = namedargs2cell( superArgs );
            self = self@LossFunction( name, superArgsCell{:}, ...
                                 type = 'Regularization', ...
                                 input = 'ZMu-ZLogVar', ...
                                 lossNets = {'Encoder'} );

        end

    end

    methods (Static)

        function loss = calcLoss( this, ZMu, ZLogVar )
            % Calculate the KL divergence
            if this.doCalcLoss
                
                loss = 0.5*sum( exp(ZLogVar) + ZMu.^2 - 1 - ZLogVar );
    
            else
                loss = 0;
            end

    end

    end

end