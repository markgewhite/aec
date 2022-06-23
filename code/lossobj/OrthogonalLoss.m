classdef OrthogonalLoss < LossFunction
    % Subclass for the Z othogonality loss (penalising correlated latent codes)
    % Code adapted from https://github.com/WangDavey/COAE

    properties

    end

    methods

        function self = OrthogonalLoss( name, superArgs )
            % Initialize the loss function
            arguments
                name                 char {mustBeText}
                superArgs.?LossFunction
            end

            superArgsCell = namedargs2cell( superArgs );
            self = self@LossFunction( name, superArgsCell{:}, ...
                                 type = 'Regularization', ...
                                 input = 'Z', ...
                                 lossNets = {'Encoder'} );

        end


        function loss = calcLoss( self, dlZ )
            % Calculate the orthogonality loss
            arguments
                self        OrthogonalLoss
                dlZ         dlarray
            end
            
            d = size( dlZ, 1 );
            dlZSq = dlVectorSq( dlZ, d );
            dlZSq = zeroDiag( dlZSq, d );
            loss = mean( dlZSq.^2, 'all' )/d;

        end


    end


end


function dlVSq = dlVectorSq( dlV, d )
    % Calculate dlV*dlV' (transpose)
    % and preserve the dlarray
    dlVSq = dlV;
    for i = 1:d
        for j = 1:d
            dlVSq(i,j) = sum( dlV(i,:).*dlV(j,:) );
        end
    end
    dlVSq = gather( dlVSq );

end


function dlV = zeroDiag( dlV, d )
    % Clear the diagonal of a dlarray
    for i = 1:d
        dlV(i,i) = 0;
    end

end