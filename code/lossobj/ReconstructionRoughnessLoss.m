classdef ReconstructionRoughnessLoss < ReconstructionLoss
    % Subclass for reconstruction roughness penalty for smoothing
    properties
        Lambda           % roughness penalty weighting
        Dilations        % numerical differentiation index step, h
    end

    methods

        function self = ReconstructionRoughnessLoss( name, superArgs, args )
            % Initialize the loss function
            arguments
                name                 char {mustBeText}
                superArgs.?ReconstructionLoss
                args.Lambda          double = 1E0
                args.Dilations       double {mustBeInteger, ...
                            mustBeGreaterThanOrEqual(args.Dilations, 1)} = 1

            end

            superArgsCell = namedargs2cell( superArgs );
            self = self@ReconstructionLoss( name, ...
                                            superArgsCell{:}, ...
                                            input = 'XGen', ...
                                            yLim = [0 0.10]);
            self.Lambda = args.Lambda;
            self.Dilations = args.Dilations;

        end


        function loss = calcLoss( self, XGen )
            % Override reconstruction loss
            arguments
                self        ReconstructionRoughnessLoss
                XGen
            end

            % calculate the loss from temporal roughness, point to point
            % iterate through dilations options and sum
            for i = 1:length(self.Dilations)
                if i==1
                    loss = self.Lambda*reconRoughnessLoss( XGen, ...
                                                           self.Scale, ...
                                                           self.Dilations(i) );
                else
                    loss = loss + self.Lambda*reconRoughnessLoss( XGen, ...
                                                           self.Scale, ...
                                                           self.Dilations(i) );
                end
            end
            
        end

    end

end