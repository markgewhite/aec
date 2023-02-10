classdef ReconstructionRoughnessLoss < ReconstructionLoss
    % Subclass for reconstruction roughness penalty for smoothing
    properties
        Lambda           % roughness penalty weighting
        Stride           % number of indices between points
    end

    methods

        function self = ReconstructionRoughnessLoss( name, superArgs, args )
            % Initialize the loss function
            arguments
                name                 char {mustBeText}
                superArgs.?ReconstructionLoss
                args.Lambda          double = 1E0
                args.Stride          double {mustBeInteger, ...
                            mustBeGreaterThanOrEqual(args.Stride, 1)} = 1
            end

            superArgsCell = namedargs2cell( superArgs );
            self = self@ReconstructionLoss( name, ...
                                            superArgsCell{:}, ...
                                            input = 'XGen', ...
                                            yLim = [0 0.20]);
            self.Lambda = args.Lambda;
            self.Stride = args.Stride;

        end


        function loss = calcLoss( self, XGen )
            % Override reconstruction loss
            arguments
                self        ReconstructionRoughnessLoss
                XGen
            end

            % calculate the loss from temporal roughness, point to point
            loss = self.Lambda*reconRoughnessLoss( XGen, self.Scale, ...
                                                   self.Stride );
            
        end

    end

end