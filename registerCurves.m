% ************************************************************************
% Function: registerCurves
% Purpose:  Perform curve registration on on functional data
%           by one or two methods:
%           - Landmark registration
%           - Continuous registration
%           Where the comparator for each method depends on either:
%           - Cross-sectional mean curve
%           - Reconstructed curve from principal components
%
% Parameters:
%       t: time span
%       XFd: smoothed curves
%       setup: smoothing settings
%       warpFd0: prior warp function (optional)
%
% Output:
%       XFdReg: registered smoothed curves
%
% ************************************************************************


function [ XFdReg, warpFd ] = registerCurves( t, XFd, type, setup, warpFd0 )

% initialise
monotonic = true;
nProcrustes = setup.nIterations;
N = size( getcoef( XFd ), 2 );

XFdReg = XFd;
if nargin < 5 || isempty( warpFd0 )
    warpT = repmat( t, N, 1 )';
else
    warpT = eval_fd( t, warpFd0 );
    warpT = max( min( warpT, t(end) ), t(1) );
end

% use a Procustes style loop
for i = 1:nProcrustes
    
    switch type
        
        case 'Landmark'
        
            % *** landmark registration ***
            
            % locate landmarks
            lm.case = setup.lmFcn( t, XFdReg, setup.lm );

            % sort into order (in necessary)
            [ lm.mean, lm.order ] = sort( lm.mean, 'ascend' );
            lm.case = lm.case( :, lm.order );

            disp(['Landmark means = ' num2str( lm.mean )]);
            disp(['Landmark SDs   = ' num2str( std( lm.case ) )]);

            wBasis = create_bspline_basis( [t(1),t(end)], ...
                                           setup.nBasis, ...
                                           setup.basisOrder, ...
                                           [ t(1) lm.mean t(end) ] );

            wFdReg = fd( zeros( setup.nBasis, 1), wBasis );
            wFdRegPar = fdPar( wFdReg, 1, setup.wLambda );

            if setup.usePC
                for j = 1:N
                    [ XFdReg, warpFd ] = landmarkreg( ...
                                    XFdReg(j), lm.case, lm.mean, ...
                                    wFdRegPar, monotonic, setup.XLambda );
                end
            else
                [ XFdReg, warpFd ] = landmarkreg( ...
                                    XFdReg, lm.case, lm.mean, ...
                                    wFdRegPar, monotonic, setup.XLambda );
            end
                                    
            if i == nProcrustes
                % final iteration
                lm = setup.lmFcn( t, XFdReg, setup.lm );
                disp(['Landmark means  = ' num2str( lm.mean )]);
                disp(['Landmark SDs    = ' num2str( std( lm.case ) )]);
            end
            
                                
        case 'Continuous'
                                    
            % *** continuous registration  ***

            wBasis = create_bspline_basis( [t(1),t(end)], ...
                                           setup.nBasis, ...
                                           setup.basisOrder );

            wFdReg = fd( zeros( setup.nBasis, 1), wBasis );
            wFdRegPar = fdPar( wFdReg, 1, setup.wLambda );

            if setup.usePC
                % use principal components
               XFdComp = reconstruct( XFdReg, setup.nPC );
               fprintf('Registering ');
               for j = 1:N
                   [ XFdReg1, warpFd1 ] = register_fd( ...
                                                  XFdComp(j), ...
                                                  XFdReg(j), ...
                                                  wFdRegPar );
                   
                   if j == 1
                       XFdRegNew = XFdReg1;
                       warpFdNew = warpFd1;
                   else
                       XFdRegNew = horzcat( XFdRegNew, XFdReg1 ); %#ok<AGROW> 
                       warpFdNew = horzcat( warpFdNew, warpFd1 ); %#ok<AGROW> 
                   end
               end
               fprintf( '\n' );
               XFdReg = XFdRegNew;
               warpFd = warpFdNew;

            else
                XFdMean = mean( XFdReg );
                [ XFdReg, warpFd ] = register_fd( XFdMean, ...
                                                  XFdReg, ...
                                                  wFdRegPar );
            end
                                   
        otherwise
            error([ 'Unrecognised registration type: ' type ]);
            
    end
  
    
    % update time warp to maintain link with original
    for j = 1:N
        % warp the previous warp
        warpT(:,j) = eval_fd( warpT(:,j), warpFd(j) );
        % separate the points evenly
        warpT(:,j) = interp1( t, warpT(:,j), t, 'spline', 'extrap' );
        % impose limits in case of over/underflow
        warpT(:,j) = max( min( warpT(:,j), t(end) ), t(1) );
    end
    warpFd = smooth_basis( t, warpT, wFdRegPar );
        
end

end


function XFdRecon= reconstruct( XFd, nPC )

    pcaXFd = pca_fd( XFd, nPC );

    XFdRecon = pcaXFd.meanfd + pcaXFd.fdhatfd;

end