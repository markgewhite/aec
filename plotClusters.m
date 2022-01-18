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

function plotClusters( ax, Z, C, CHat, compact )

if nargin>=4
    if isa( CHat, 'logical' )
        inclPredictedClass = false;
        compact = CHat;
    else
        inclPredictedClass = true;
        if nargin==4
            compact = false;
        end
    end
else
    inclPredictedClass = false;
    compact = false;
end

C = categorical( C );
class = sort( unique( C ) );

cla( ax, 'reset' );
hold( ax, 'on' );

if compact
    dotSize = 10;
    dotSize2 = 4;
else
    dotSize = 40;
    dotSize2 = 10;
end

% plot true classes (large dots)
ax.ColorOrderIndex = 1;
for i = 1:length( class )
    idx = (C==class(i));
    scatter( ax, Z(idx,1), Z(idx,2), dotSize, 'filled' );
end

if inclPredictedClass
    % plot estimated classes (small dots on top)
    CHat = categorical( CHat );
    ax.ColorOrderIndex = 1;
    for i = 1:length( class )
        idx = (CHat==class(i));
        scatter( ax, Z(idx,1), Z(idx,2), dotSize2, 'filled' );
    end
end

if ~compact
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
end

hold( ax, 'off' );

if ~compact
    legend( ax, class, 'Location', 'Best' );
end

end
