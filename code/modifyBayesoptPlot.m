% Modify Bayesopt 1D plots

fig = figure(1);

ax = fig.CurrentAxes;

fig.Position(3) = 474;
fig.Position(4) = 432;

xlabel( ax, 'Z Dim' );
ylabel( ax, 'Proportion classified in error' );
title( ax, 'Test Classification Loss for Auxiliary Logistic Model' );
ax.LineWidth = 1;
ax.TickDir = 'out';
ax.FontSize = 14;

ylim( ax, [0 0.15] );
ax.YAxis.Exponent = 0;
ax.YAxis.TickLabelFormat = '%.2f';
