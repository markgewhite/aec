function [F, prc, ZQ, offsets ] = calcPDP( self, dlZ, args )
    % Partial Dependence Plot
    % For latent component generation and the auxiliary model
    arguments
        self                RepresentationModel
        dlZ                 {mustBeA( dlZ, {'dlarray', 'double'} )}
        args.sampling       char ...
                            {mustBeMember(args.sampling, ...
                            {'Regular', 'Component'} )} = 'Regular'
        args.nSample        double {mustBeInteger} = 20
        args.maxObs         double = 1000
        args.modelFcn       function_handle
        args.modelFcnArgs   cell = []
    end
    
    if isa( dlZ, 'dlarray' )
        % convert to double for quantiles, sort and other functions
        Z = double(extractdata( dlZ ));
    else
        if size(dlZ,1) ~= self.ZDim
            % transpose into standard dimensions:
            % 1st=ZDim and 2nd=observations
            dlZ = dlZ';
        end
        Z = dlZ;
    end

    nObs = size( dlZ, 2 );
    if nObs > args.maxObs
        % data too large - subsample
        subset = randsample( nObs, args.maxObs );
        dlZ = dlZ( :, subset );
        Z = Z( :, subset );
    end

    % generate the quantiles and required Z values
    K = args.nSample;
    switch args.sampling
        case 'Regular'
            prc = linspace( 0, 1, K+1 );
        case 'Component'
            prc = linspace( 0.05, 0.95, self.NumCompLines );
    end
    offsets = norminv( prc ); % z-scores
    ZQ = quantile( Z, prc, 2 );
    K = size(ZQ, 2);

    % take the mean to obtain reference values
    dlZM = mean( dlZ, 2 );

    for d = 1:self.ZDim

        for k = 1:K
   
            % set all elements to the median
            dlZC = dlZM;
            % set the dth element to the kth value
            dlZC(d) = ZQ( d, k );
            if isempty( args.modelFcnArgs )
                YHat = args.modelFcn( self, dlZC );
            else
                YHat = args.modelFcn( self, dlZC, args.modelFcnArgs{:} );
            end

            if d==1
                % allocate arrays knowing the size of YHat
                if size(YHat,3)==1
                    FDim(1) = size( YHat, 1 );
                    FDim(2) = 1;
                    F = zeros( self.ZDim, K, FDim(1) );
                else
                    FDim(1) = size( YHat, 1 );
                    FDim(2) = size( YHat, 2 );
                    F = zeros( self.ZDim, K, FDim(1), FDim(2) );
                end
            end

            % assign YHat to response array
            F( d, k, : ) = YHat;

        end

    end

end
