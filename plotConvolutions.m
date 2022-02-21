% ************************************************************************
% Function: plotConvolutions
%
% Plot convolutions 
%
% Parameters:
%           conv    : convolutions
%           
% Outputs:
%
% ************************************************************************

function plotConvolutions( ax, tSpan, conv, Y, dilation, padding )

len = length( conv{1} );
idxRng = -padding:dilation:len*dilation;
idxRng = idxRng + fix( len/2 ); % midpoint
valid = idxRng>=1 & idxRng<=length(tSpan);
idxRng = idxRng( valid );

t = linspace( tSpan(1), tSpan(end), len );

colour = [ 0.00 0.45 0.74; 0.85 0.33 0.10 ];

hold( ax, 'off' );
plot( ax, t, conv{1}, 'Color', colour(Y(1),:) );

hold( ax, 'on' );
for i = 2:length(conv)
    plot( ax, t, conv{i}, 'Color', colour(Y(i),:) );
end

hold( ax, 'off' );

end