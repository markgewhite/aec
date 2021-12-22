% *************************************
% Update the Z distributions plot
% *************************************

function plotZDist( ax, Z, name, stdize )

if nargin<4
    stdize = false;
end
if stdize
    Z = (Z-mean(Z,2))./std(Z,[],2);
    xLbl = 'Std Z';
else
    xLbl = 'Z';
end

nPts = 101;
nCodes = size( Z, 1 );


hold( ax, 'off');
for i = 1:nCodes
    pdZ = fitdist( Z(i,:)', 'Kernel', 'Kernel', 'epanechnikov' );
    ZMin = prctile( Z(i,:), 0.0001 );
    ZMax = prctile( Z(i,:), 99.9999 );
    ZPts = ZMin : (ZMax-ZMin)/(nPts-1) : ZMax;
    Y = pdf( pdZ, ZPts );
    Y = Y/sum(Y);
    plot( ax, ZPts, Y, 'LineWidth', 1 );
    hold( ax, 'on' );
end
hold( ax, 'off');

ylim( ax, [0 0.1] );

title( ax, name );
xlabel( ax, xLbl );
ylabel( ax, 'Q(Z)' );


drawnow;

end