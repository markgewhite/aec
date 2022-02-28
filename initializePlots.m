% ************************************************************************
% Function: initializePlots
%
% Initialise the figures
%
% Parameters:
%           config : data setup structure
%           
% Outputs:
%           ax : axes objects
%
% ************************************************************************

function ax = initializePlots( nCodes, nChannels )

figure(3);
ax.ae.pred = subplot( 2,2,1 );
ax.pca.pred = subplot( 2,2,2 );
ax.ae.cls = subplot( 2,2,3 );
ax.pca.cls = subplot( 2,2,4 );

figure(4);
ax.ae.distZTrn = subplot( 2,2,1 );
ax.pca.distZTrn = subplot( 2,2,2 );
ax.ae.distZTst = subplot( 2,2,3 );
ax.pca.distZTst = subplot( 2,2,4 );

ax.ae.comp = gobjects( nCodes, nChannels );
[ rows, cols ] = sqdim( nCodes );

for j = 1:nChannels
    figure(4+j);
    for i = 1:nCodes
        ax.ae.comp(i,j) = subplot( rows, cols, i );
    end
end

end