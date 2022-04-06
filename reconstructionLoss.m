% ************************************************************************
% Class: reconstructionLoss
%
% Subclass for reconstruction error
%
% ************************************************************************

classdef reconstructionLoss < lossFcn

    properties

    end

    methods

        function self = reconstructionLoss( name, superArgs )
            % Initialize the loss function
            arguments
                name                 char {mustBeText}
                superArgs.?lossFcn
            end

            superArgsCell = namedargs2cell( superArgs );
            self = self@lossFcn( name, superArgsCell{:}, ...
                                 type = 'Reconstruction' );

        end

    end

    methods (Static)

    function loss = calcLoss( self, X, XHat )
        % Calculate the reconstruction loss
        if self.doCalcLoss
            loss = mse( X, XHat );
        else
            loss = 0;
        end

    end

    end

end