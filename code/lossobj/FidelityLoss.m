classdef FidelityLoss < LossFunction
    % Subclass for the Z fidelity loss 
    % Code adapted from (C) Laurens van der Maaten, 2008
    % Maastricht University

    properties

    end

    methods

        function self = FidelityLoss( name, superArgs )
            % Initialize the loss function
            arguments
                name                 char {mustBeText}
                superArgs.?LossFunction
            end

            superArgsCell = namedargs2cell( superArgs );
            self = self@LossFunction( name, superArgsCell{:}, ...
                                 type = 'Regularization', ...
                                 input = 'P-Z', ...
                                 lossNets = {'Encoder', 'Decoder'} );

        end


        function loss = calcLoss( self, dlP, dlZ )
            % Calculate the fidelity loss
            arguments
                self        FidelityLoss
                dlP         dlarray
                dlZ         dlarray
            end
            
            % center Z
            dlZ = dlZ - mean(dlZ);
        
            % remove dimension labels to enable calculations below
            dlZ = stripdims(dlZ);
        
            % compute joint probability that point i and j are neighbors
            sum_dlZ = sum(dlZ .^ 2);
        
            % Student-t distribution
            dlN = 1 ./ (1 + sum_dlZ + sum_dlZ' - 2*(dlZ')*dlZ);
            
            % set diagonal to zero
            dlN(1:size(dlZ,2)+1:end) = 0;
            
            % normalize to get probabilities
            dlQ = max(dlN ./ sum(dlN(:)), realmin);

            % compute loss base on the KL divergence
            loss = -sum( (dlP - dlQ).*dlN, 'all' );

        end


    end


end
