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
                
            dlVSq = dlVectorSq( dlZ );
            dlVSqDiag = dlDiag( dlVSq );
            loss = 0.01*mean( (dlVSq - dlVSqDiag).^2, 'all' );

        end


    end


end


function dlVSq = dlVectorSq( dlV )
    % Calculate dlV*dlV' (transpose)
    % and preserve the dlarray
    r = size( dlV, 1 );
    dlVSq = dlarray( zeros(r,r), 'CB' );
    for i = 1:r
        for j = 1:r
            dlVSq(i,j) = sum( dlV(i,:).*dlV(j,:) );
        end
    end

end


function dlVDiag = dlDiag( dlV )
    % Extract the diagonal of a dlarray
    r = size( dlV, 1 );
    dlVDiag = dlarray( zeros(r,r), 'CB' );
    for i = 1:r
        dlVDiag(i,i) = dlV(i,i);
    end

end