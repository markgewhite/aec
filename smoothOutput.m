% ************************************************************************
% Function: smoothOutput
%
% Perform smoothing on the decoder output
%
% Parameters:
%           X       : initial X
%           type    : type of smoothing
%           window  : smoothing window
%           
% Outputs:
%           XS      : smoothed X
%
% ************************************************************************

function XS = smoothOutput( X, type, window )

[ nPts, nBatch ] = size( X );

XS = X;
switch type
    case 'MovAvg'
        pad = fix( window/2 );
        XP = [ zeros( pad, nBatch ); X; zeros( pad, nBatch ) ];
        for i = 1:nPts
            XS( i, : ) = mean( XP( i:i+window-1, : ) );
        end

    case 'Gaussian' % 11-point window hardwired
        pad = 5;
        XP = [ zeros( pad, nBatch ); X; zeros( pad, nBatch ) ];
        w = [0.044 0.135 0.325 0.607 0.883 1.000 ...
             0.883 0.607 0.325 0.135 0.044];
        for i = 1:nPts
            XS( i, : ) = w*XP( i:i+10, : )/4.988;
        end
        
end

end
