classdef ReconstructionRoughnessLoss < ReconstructionLoss
    % Subclass for reconstruction roughness penalty for smoothing
    properties
        Lambda           % roughness penalty weighting
    end

    methods

        function self = ReconstructionRoughnessLoss( name, superArgs, args )
            % Initialize the loss function
            arguments
                name                 char {mustBeText}
                superArgs.?ReconstructionLoss
                args.Lambda          double = 1E0
            end

            superArgsCell = namedargs2cell( superArgs );
            self = self@ReconstructionLoss( name, ...
                                            superArgsCell{:}, ...
                                            input = 'XGen', ...
                                            yLim = [0 0.20]);
            self.Lambda = args.Lambda;

        end


        function loss = calcLoss( self, XGen )
            % Override reconstruction loss
            arguments
                self        ReconstructionRoughnessLoss
                XGen
            end

            % calculate the loss from temporal variance, point to point
            loss = self.Lambda*reconRoughnessLoss( XGen, self.Scale );
            
        end

    end

end