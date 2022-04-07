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
        kernel     % type of kernel to use
        scale      % kernel scale 
        baseType   % target distribution type
    end

    methods

        function self = wassersteinLoss( name, superArgs, args )
            % Initialize the loss function
            arguments
                name                 char {mustBeText}
                superArgs.?lossFunction
                args.kernel       char ...
                    {mustBeMember( args.kernel, {'RBF', 'IMQ'} )} = 'IMQ'
                args.scale        double = 2
                args.baseType     char ...
                    {mustBeMember( args.baseType, ...
                        {'Normal', 'Sphere', 'Uniform', 'ZDim'} )} = 'Normal'
            end

            superArgsCell = namedargs2cell( superArgs );
            self = self@lossFunction( name, superArgsCell{:}, ...
                                 type = 'Regularization', ...
                                 input = 'Z-ZHat' );
            self.kernel = args.kernel;
            self.scale = args.scale;
            self.baseType = args.baseType;

        end

    end

    methods (Static)

        function loss = calcLoss( self, dlZP, dlZQ )
            % Calculate the MMD loss
            arguments
                self
                dlZP  dlarray  % real distribution
                dlZQ  dlarray  % generated distribution
            end

            if self.doCalcLoss

                % switch to non-dlarrays for faster processing
                % do I need to do this?
                ZQ = double(extractdata( dlZQ ) )';
                ZP = double(extractdata( dlZP ) )';
   
                switch self.kernel
                    case 'RBF'
                        loss = mmdLossRBF( ZP, ZQ );
                    case 'IMQ'
                        loss = mmdLossIMQ( ZP, ZQ, ...
                                        self.scale, self.baseType );
                end
            else
                loss = 0;
            end
    
        end


        function loss = mmdLossRBF( ZP, ZQ )
            % Calculate the MMD loss using a RBF kernel

            nObs = size( ZQ, 1 );

            [distPP, distQQ, distPQ] = mmdDist( ZP, ZQ );

            % median heuristic for the sigma^2 of Gaussian kernel
            % modified from tolstikhin/wae:
            % it seems to make more sense to take the mean of both
            % and then use it to standardize the distances;
            % originally, it doubled these combined median values
            % - could this be why their RBF did not work so well?
            sigma2_k = median( distPQ, 'all' ); 
            sigma2_k = sigma2_k + median( distQQ, 'all' ); 

            res1 = exp( -distQQ/sigma2_k );
            res1 = res1 + exp( -distPP/sigma2_k );
            res1 = res1*(ones(nObs) - eye(nObs));
            res1 = sum( res1, 'all' )/(nObs*nObs-nObs);

            res2 = exp( -(distPQ/2)/sigma2_k );
            res2 = sum( res2, 'all' )*2/(nObs*nObs);
            
            loss = res1 - res2;

        end


        function loss = mmdLossIMQ( ZP, ZQ, scale, baseType )
            % Calculate the MMD loss using an IMQ kernel

            sigma2_p = scale^2;
            [ nObs, ZDim ] = size( ZQ );

            [distPP, distQQ, distPQ] = mmdDist( ZP, ZQ );

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


        function [distPP, distQQ, distPQ] = mmdDist( ZP, ZQ )
            % Calculate the maximum mean distances between points
            % Between P & P, Q & Q and Q & P

            norms_pz = sum( ZP.^2, 2 );
            dotprods_pz = ZP.*PZ';
            distances_pz = norms_pz + norms_pz' - 2*dotprods_pz;
            distPP = pdist2( ZP, ZP ).^2;
            
            norms_qz = sum(ZQ.^2);
            dotprods_qz = ZQ.*ZQ';
            distances_qz = norms_qz + norms_qz' - 2*dotprods_qz;
            distQQ = pdist2( ZQ, ZQ ).^2;
            
            dotprods = ZQ.*ZP';
            distances = norms_qz + norms_pz' - 2*dotprods;
            distPQ = pdist2( ZQ, ZP ).^2;

        end


    end

end