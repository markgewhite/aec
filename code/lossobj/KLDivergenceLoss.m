classdef KLDivergenceLoss < LossFunction
    % Subclass for Kullback-Leibler divergence loss

    properties
        Beta        % loss scaling factor
    end

    methods

        function self = KLDivergenceLoss( name, superArgs, args )
            % Initialize the loss function
            arguments
                name                 char {mustBeText}
                superArgs.?LossFunction
                args.Beta            double = 0.1
            end

            superArgsCell = namedargs2cell( superArgs );
            self = self@LossFunction( name, superArgsCell{:}, ...
                                 Type = 'Regularization', ...
                                 Input = {'dlMu', 'dlLogVar'}, ...
                                 LossNets = {'Encoder'}, ...
                                 YLim = [0 0.1]);

            self.Beta = args.Beta;

        end


        function loss = calcLoss( self, dlMu, dlLogVar )
            % Calculate the KL divergence
            arguments
                self        KLDivergenceLoss
                dlMu        dlarray
                dlLogVar    dlarray
            end

            if ~isempty( dlMu ) && ~isempty( dlLogVar )
                loss = -0.5*sum( 1 + dlLogVar - dlMu.^2 - exp(dlLogVar) );
                loss = self.Beta*mean( loss );
            else
                loss = 0;
            end

        end

    end

end