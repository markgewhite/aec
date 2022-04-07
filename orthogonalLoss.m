% ************************************************************************
% Class: orthogonalLoss
%
% Subclass for the Z othogonality loss (penalising correlated latent codes)
%
% Code adapted from https://github.com/WangDavey/COAE
%
% ************************************************************************

classdef orthogonalLoss < lossFunction

    properties

    end

    methods

        function self = orthogonalLoss( name, superArgs )
            % Initialize the loss function
            arguments
                name                 char {mustBeText}
                superArgs.?lossFunction
            end

            superArgsCell = namedargs2cell( superArgs );
            self = self@lossFunction( name, superArgsCell{:}, ...
                                 type = 'Regularization', ...
                                 input = 'Z' );

        end

    end

    methods (Static)

        function loss = calcLoss( self, dlZ )
            % Calculate the orthogonality loss
            if self.doCalcLoss
                
                orth = dlVectorSq( dlZ );
                loss.orth = ...
                    sqrt(sum(orth.^2,'all') - sum(diag(orth).^2))/ ...
                                sum(dlZ.^2,'all');
    
            else
                loss = 0;
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

    end


end