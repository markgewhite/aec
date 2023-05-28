classdef WassersteinLoss < LossFunction
    % Subclass for wasserstein loss based on the 
    % Maximum Mean Discrepancy distance between samples
    
    % Code adapted from https://github.com/tolstikhin/wae

    properties
        Kernel        % type of kernel to use
        Scale         % kernel scale 
        BaseType      % kernel distribution
        Distribution  % target distribution
    end

    methods

        function self = WassersteinLoss( name, superArgs, args )
            % Initialize the loss function
            arguments
                name                 char {mustBeText}
                superArgs.?LossFunction
                args.Kernel          char ...
                    {mustBeMember( args.Kkernel, {'RBF', 'IMQ'} )} = 'IMQ'
                args.Scale           double = 2
                args.BaseType        char ...
                    {mustBeMember( args.BaseType, ...
                        {'Normal', 'Sphere', 'Uniform', 'ZDim'} )} = 'Normal'
                args.Distribution    char ...
                    {mustBeMember( args.Distribution, ...
                        {'Gaussian', 'Categorical'} )} = 'Gaussian'
            end

            superArgsCell = namedargs2cell( superArgs );
            self = self@LossFunction( name, superArgsCell{:}, ...
                                 Type = 'Regularization', ...
                                 Input = {'dlZ'}, ...
                                 LossNets = {'Encoder'} );
            
            self.Kernel = args.Kernel;
            self.Scale = args.Scale;
            self.BaseType = args.BaseType;
            self.Distribution = args.Distribution;

        end

        function loss = calcLoss( self, dlZQ )
            % Calculate the MMD loss
            arguments
                self
                dlZQ  dlarray  % generated distribution
            end

            [ ZDim, batchSize ] = size( dlZQ );

            % generate a target distribution
            switch self.Distribution
                case 'Gaussian'
                    ZP = randn( ZDim, batchSize );
                case 'Categorical'
                    ZP = randi( ZDim, batchSize );
            end
            dlZP = dlarray( ZP, 'CB' );

            [dlDistPP, dlDistQQ, dlDistPQ] = mmd( dlZP, dlZQ ); 

            switch self.Kernel
                case 'RBF'
                    loss = mmdLossRBF( dlDistPP, dlDistQQ, dlDistPQ, ...
                                       batchSize );
                case 'IMQ'
                    loss = mmdLossIMQ( dlDistPP, dlDistQQ, dlDistPQ, ...
                                       batchSize, ZDim, ...
                                       self.Scale, self.BaseType );
            end

        end

    end


end

function loss = mmdLossRBF( dlDistPP, dlDistQQ, dlDistPQ, nObs )
    % Calculate the MMD loss using a RBF kernel

    % median heuristic for the sigma^2 of Gaussian kernel
    % modified from tolstikhin/wae:
    % it seems to make more sense to take the mean of both
    % and then use it to standardize the distances;
    % originally, it doubled these combined median values
    % - could this be why their RBF did not work so well?
    sigma2_k = (median( dlDistPQ, 'all' ) + median( dlDistQQ, 'all' ))/2; 

    res1 = exp( -dlDistQQ/sigma2_k );
    res1 = res1 + exp( -dlDistPP/sigma2_k );
    res1 = res1.*(ones(nObs) - eye(nObs));
    res1 = sum( res1, 'all' )/(nObs*nObs-nObs);

    res2 = exp( -distPQ/sigma2_k );
    res2 = sum( res2, 'all' )*2/(nObs*nObs);
    
    loss = 10*(res1 - res2);

end


function loss = mmdLossIMQ( dlDistPP, dlDistQQ, dlDistPQ, ...
                            nObs, ZDim, scale, baseType )
    % Calculate the MMD loss using an IMQ kernel

    sigma2_p = scale^2;

    switch baseType
        case 'Normal'
            Cbase = 2*ZDim*sigma2_p;
        case 'Sphere'
            Cbase = 2;
        case 'Uniform'
            Cbase = ZDim;
    end

    loss = 0;
    scale = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0];
    mask = ones(nObs) - eye(nObs);

    for i = 1:length(scale)
        
        C = Cbase*scale(i);
        
        res1 = C./(C + dlDistQQ);
        res1 = res1 + C./(C + dlDistPP);
        res1 = res1.*mask;
        res1 = sum( res1, 'all' )/(nObs*nObs-nObs);

        res2 = C./(C + dlDistPQ);
        res2 = sum( res2, 'all' )*2/(nObs*nObs);

        loss = loss + res1 - res2;

    end

end


function [ dlPPDist, dlQQDist, dlPQDist ] = mmd( dlP, dlQ )
    % Calculate the maximum mean squared distance between points
    arguments
        dlP     dlarray
        dlQ     dlarray
    end
    
    dlPPNorm = sum( dlP.^2 );
    dlPPDotProd = dlMultiply( dlP, dlP );
    dlPPDist = dlPPNorm + dlTranspose( dlPPNorm ) - 2*dlPPDotProd;

    dlQQNorm = sum( dlQ.^2 );
    dlQQDotProd = dlMultiply( dlQ, dlQ );
    dlQQDist = dlQQNorm + dlTranspose( dlQQNorm ) - 2*dlQQDotProd;
    
    dlPQDotProd = dlMultiply( dlP, dlQ );
    dlPQDist = dlQQNorm + dlTranspose( dlPPNorm ) - 2*dlPQDotProd;

end


function dlM = dlMultiply( dlV, dlW )
    % Calculate dlV*dlV' (transpose)
    % and preserve the dlarray
    [ r, c ]= size( dlV );
    dlM = dlarray( zeros(c,c), 'CB' );
    for i = 1:c
        for j = 1:c
            dlM(i,j) = sum( dlV(:,i).*dlW(:,j) );
        end
    end

end



function dlVT = dlTranspose( dlV )
    % Take the transpose of a dlarray
    arguments
        dlV     dlarray
    end

    d = size( dlV );

    dlVT = dlarray( zeros( d(2), d(1) ) );
    for i = 1:d(2)
        for j = 1:d(1)
            dlVT(i,j) = dlV(j,i);
        end
    end

end

