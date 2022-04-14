% ************************************************************************
% Class: wassersteinLoss
%
% Subclass for wasserstein loss based on the Maximum Mean Discrepancy 
% distance between samples
%
% Code adapted from https://github.com/tolstikhin/wae
%
% ************************************************************************

classdef wassersteinLoss < lossFunction

    properties
        kernel        % type of kernel to use
        scale         % kernel scale 
        baseType      % kernel distribution
        distribution  % target distribution
    end

    methods

        function self = wassersteinLoss( name, superArgs, args )
            % Initialize the loss function
            arguments
                name                 char {mustBeText}
                superArgs.?lossFunction
                args.kernel          char ...
                    {mustBeMember( args.kernel, {'RBF', 'IMQ'} )} = 'IMQ'
                args.scale           double = 2
                args.baseType        char ...
                    {mustBeMember( args.baseType, ...
                        {'Normal', 'Sphere', 'Uniform', 'ZDim'} )} = 'Normal'
                args.distribution    char ...
                    {mustBeMember( args.distribution, ...
                        {'Gaussian', 'Categorical'} )} = 'Gaussian'
            end

            superArgsCell = namedargs2cell( superArgs );
            self = self@lossFunction( name, superArgsCell{:}, ...
                                 type = 'Regularization', ...
                                 input = 'Z', ...
                                 lossNets = {'encoder'} );
            
            self.kernel = args.kernel;
            self.scale = args.scale;
            self.baseType = args.baseType;
            self.distribution = args.distribution;

        end

        function loss = calcLoss( self, dlZQ )
            % Calculate the MMD loss
            arguments
                self
                dlZQ  dlarray  % generated distribution
            end

            [ ZSize, batchSize ] = size( dlZQ );

            % switch to non-dlarrays for faster processing
            % do I need to do this?
            ZQ = double(extractdata( dlZQ ) )';

            % generate a target distribution
            switch self.distribution
                case 'Gaussian'
                    ZP = randn( ZSize, batchSize )';
                case 'Categorical'
                    ZP = randi( ZSize, batchSize )';
            end

            switch self.kernel
                case 'RBF'
                    loss = mmdLossRBF( ZP, ZQ );
                case 'IMQ'
                    loss = mmdLossIMQ( ZP, ZQ, self.scale, self.baseType );
            end

        end

    end


end

function loss = mmdLossRBF( ZP, ZQ )
    % Calculate the MMD loss using a RBF kernel

    nObs = size( ZQ, 1 );

    % calculate the maximum mean distances between points
    distPP = pdist2( ZP, ZP ).^2;
    distQQ = pdist2( ZQ, ZQ ).^2;
    distPQ = pdist2( ZQ, ZP ).^2;

    % median heuristic for the sigma^2 of Gaussian kernel
    % modified from tolstikhin/wae:
    % it seems to make more sense to take the mean of both
    % and then use it to standardize the distances;
    % originally, it doubled these combined median values
    % - could this be why their RBF did not work so well?
    sigma2_k = (median( distPQ, 'all' ) + median( distQQ, 'all' ))/2; 

    res1 = exp( -distQQ/sigma2_k );
    res1 = res1 + exp( -distPP/sigma2_k );
    res1 = res1.*(ones(nObs) - eye(nObs));
    res1 = sum( res1, 'all' )/(nObs*nObs-nObs);

    res2 = exp( -distPQ/sigma2_k );
    res2 = sum( res2, 'all' )*2/(nObs*nObs);
    
    loss = 10*(res1 - res2);

end


function loss = mmdLossIMQ( ZP, ZQ, scale, baseType )
    % Calculate the MMD loss using an IMQ kernel

    sigma2_p = scale^2;
    [ nObs, ZDim ] = size( ZQ );

    % calculate the maximum mean distances between points
    distPP = pdist2( ZP, ZP ).^2;
    distQQ = pdist2( ZQ, ZQ ).^2;
    distPQ = pdist2( ZQ, ZP ).^2;

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
        
        res1 = C./(C + distQQ);
        res1 = res1 + C./(C + distPP);
        res1 = res1.*mask;
        res1 = sum( res1, 'all' )/(nObs*nObs-nObs);

        res2 = C./(C + distPQ);
        res2 = sum( res2, 'all' )*2/(nObs*nObs);

        loss = loss + res1 - res2;

    end

end


