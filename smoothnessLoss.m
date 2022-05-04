% ************************************************************************
% Class: smoothnessLoss
%
% Subclass for the loss functions computed from AE components
%
% ************************************************************************

classdef smoothnessLoss < lossFunction

    properties
    end

    methods

        function self = smoothnessLoss( name, superArgs )
            % Initialize the loss function
            arguments
                name                 char {mustBeText}
                superArgs.?lossFunction
            end

            superArgsCell = namedargs2cell( superArgs );
            self = self@lossFunction( name, superArgsCell{:}, ...
                                 type = 'Output', ...
                                 input = 'XHat', ...
                                 lossNets = {'encoder', 'decoder'} );

        end


        function loss = calcLoss( self, dlXHat )
            % Calculate the component loss
            arguments
                self
                dlXHat  dlarray  % output
            end

            % calculate the smoothed curve
            XHat = double(extractdata( dlXHat));
            XHatSmth = smoothdata( XHat, 'Gaussian', 10 );

            loss = mean( (XHat - XHatSmth).^2, 'all' );

        end

    end

end


