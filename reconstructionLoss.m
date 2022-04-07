% ************************************************************************
% Class: reconstructionLoss
%
% Subclass for reconstruction error
%
% ************************************************************************

classdef reconstructionLoss < lossFunction

    properties

    end

    methods

        function self = reconstructionLoss( name, superArgs )
            % Initialize the loss function
            arguments
                name                 char {mustBeText}
                superArgs.?lossFunction
            end

            superArgsCell = namedargs2cell( superArgs );
            self = self@lossFunction( name, superArgsCell{:}, ...
                                 type = 'Reconstruction', ...
                                 input = 'X-XHat' );

        end

    end

    methods (Static)

    function loss = calcLoss( X, XHat )
        % Calculate the reconstruction loss
        if self.doCalcLoss
            loss = mse( X, XHat );
        else
            loss = 0;
        end

    end

    end

end