% ************************************************************************
% Function: plotClusters
%
% Plot assigned classes on 2D grid
%
% Parameters:
%           ax          : plot axes
%           Z           : 2D encodings
%           C           : true classes
%           CHat        : predicted classes
% ************************************************************************

function plotClusters( ax, Z, C, CHat )

C = categorical( C );
CHat = categorical( CHat );

class = sort( unique( C ) );

cla( ax, 'reset' );
hold( ax, 'on' );

% plot true classes (large dots)
ax.ColorOrderIndex = 1;
for i = 1:length( class )
    idx = (C==class(i));
    scatter( ax, Z(idx,1), Z(idx,2), 40, 'filled' );
end

% plot estimated classes (small dots on top)
ax.ColorOrderIndex = 1;
for i = 1:length( class )
    idx = (CHat==class(i));
    scatter( ax, Z(idx,1), Z(idx,2), 10, 'filled' );
end

% then add the centroids on top
for i = 1:length( class )
    idx = (C==class(i));
    text( ax, mean( Z(idx,1) ), mean( Z(idx,2) ), ...
              class(i), ...
              'HorizontalAlignment', 'center', ...
              'FontWeight', 'bold', ...
              'FontSize', 10, ...
              'Color', [0 0 0] );
end

hold( ax, 'off' );

legend( ax, class, 'Location', 'Best' );

end
