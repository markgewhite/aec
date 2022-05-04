% ************************************************************************
% Class: smoothnessLoss
%
% Subclass for the loss functions computed from AE components
%
% ************************************************************************

classdef smoothnessLoss < lossFunction

    properties
        window
    end

    methods

        function self = smoothnessLoss( name, args, superArgs )
            % Initialize the loss function
            arguments
                name                 char {mustBeText}
                args.window          double ...
                    {mustBeInteger,mustBePositive} = 10
                superArgs.?lossFunction
            end

            superArgsCell = namedargs2cell( superArgs );
            self = self@lossFunction( name, superArgsCell{:}, ...
                                 type = 'Output', ...
                                 input = 'XHat', ...
                                 lossNets = {'encoder', 'decoder'} );

            self.window = args.window;

        end


        function loss = calcLoss( self, dlXHat )
            % Calculate the component loss
            arguments
                self        smoothnessLoss
                dlXHat      dlarray  % output
            end

            % calculate the smoothed curve
            XHat = double(extractdata( dlXHat));
            XHatSmth = smoothdata( XHat, 'Gaussian', self.window );

            loss = mean( (XHat - XHatSmth).^2, 'all' );

        end

    end

end


