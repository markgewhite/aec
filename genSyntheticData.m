% ************************************************************************
% Function: genFunctionalData
%
% Generate synthetic functional data 
% based originally on the method proposed by Hsieh et al. (2021).
%
% Enhanced with option to have multiple basis levels.
% The number of levels is specified if basis is a cell array
% 
% Parameters:
%           nObs        : number of observations per class (vector)
%           nDim        : number of dimensions
%           setup
%             .tSpan    : time span vector
%             .nLevels  : number of time span levels
%             .mu       : array of template magnitudes
%             .sigma    : array of template magnitude variances
%             .tau      : warping variance
%           
% Outputs:
%           Z     : generated data points
%
% ************************************************************************

function Z = genSyntheticData( nObs, nDim, setup )

% initialise the number of points across multiple layers
% allow extra space either end for extrapolation when time warping
% the time domains are twice as long
nLevels = length( setup.ratio );

nPts = zeros( nLevels, 1 );
tSpan = cell( nLevels, 1 );
range = [ setup.tSpan(1), setup.tSpan(end) ];
extra = 0.5*(range(2)-range(1));
dt = setup.tSpan(2)-setup.tSpan(1);

for j = 1:nLevels
    nPts(j) = 2*((length( setup.tSpan )-1)/setup.ratio(j))+1;
    tSpan{j} = linspace( range(1)-extra, range(2)+extra, nPts(j) )';
end

tWarp0 = tSpan{ setup.warpLevel };

% initialise the template array across levels
template = zeros( nPts(1), nDim, nLevels );

% initialise the array holding the generated data
Z = zeros( length( setup.tSpan ), sum(nObs), nDim );

% define the common template shared by all classes
for j = setup.sharedLevel:nLevels
        template( :,:,j ) = interpRandSeries( tSpan{j}, tSpan{1}, ...
                                                 nPts(j), nDim, 2 );
end

a = 0;
for c = 1:length(nObs)

    % generate random template function coefficients
    % with covariance between the series elements
    % interpolating to the base layer (1)
    for j = 1:setup.sharedLevel-1
        template( :,:,j ) = interpRandSeries( tSpan{j}, tSpan{1}, ...
                                                 nPts(j), nDim, 2 );
    end
   
    for i = 1:nObs(c)

        a = a+1;

        % vary the template function across levels
        sample = zeros( nPts(1), nDim );
        for j = 1:nLevels 
            sample = sample + (setup.mu(j) + setup.sigma(j)*randn(1,1)) ...
                                * template( :,:,j );
        end

        % introduce noise
        sample = sample + setup.eta*randn( nPts(1), nDim );

        % warp the time domain at the top level, ensuring monotonicity
        % and avoiding excessive curvature by constraining the gradient
        monotonic = false;
        excessCurvature = false;
        while ~monotonic || excessCurvature
            % generate a time warp series based on the top-level 
            tWarp = tWarp0 + setup.tau*randSeries( 1, length(tWarp0) )';
            % interpolate so it fits the initial level
            tWarp = interp1( tWarp0, tWarp, tSpan{1}, 'spline' );
            % check constraints
            grad = diff( tWarp )/dt;
            monotonic = all( grad>0 );
            excessCurvature = any( grad<0.2 );
        end

        % interpolate the coefficients to the warped time points
        Z( :, a, : ) = interp1( tWarp, sample, setup.tSpan, 'spline' );
               
    end

end

end
