% ************************************************************************
% Function: plotKernel
%
% Plot kernel 
%
% Parameters:
%           
%           
% Outputs:
%
% ************************************************************************

function plotKernel( ax, tSpan, i, kernels )

len = kernels.lengths(i);
wStart = sum( kernels.lengths(1:i-1) );
w = kernels.weights( wStart: wStart+len-1 );

t = linspace( tSpan(1), tSpan(end), len );

plot( ax, w, 'LineWidth', 2 );

end