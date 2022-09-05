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
                args.Beta            double = 1
            end

            superArgsCell = namedargs2cell( superArgs );
            self = self@LossFunction( name, superArgsCell{:}, ...
                                 type = 'Regularization', ...
                                 input = 'Z', ...
                                 lossNets = {'Encoder'} );

            self.Beta = args.Beta;

        end


        function loss = calcLoss( self, dlZ )
            % Calculate the KL divergence
            arguments
                self        KLDivergenceLoss
                dlZ         dlarray
            end

            dlMu = mean( dlZ );
            dlLogVar = log( var( dlZ ) );

            loss = -0.5*sum( 1 + dlLogVar - dlMu.^2 - exp(dlLogVar) );
            loss = self.Beta*mean( loss );

        end

    end

end