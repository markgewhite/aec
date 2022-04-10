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
                                 input = 'X-XHat', ...
                                 lossNets = {'encoder', 'decoder'} );

        end

    end

    methods (Static)

    function loss = calcLoss( X, XHat )
        % Calculate the reconstruction loss
        loss = mean( (X-XHat).^2, 'all' );

    end

    end

end