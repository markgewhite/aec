% ************************************************************************
% Function: initializeData
%
% Initialise the data setup
%
% Parameters:
%           
% Outputs:
%           X     : data as cell array, variable length (fine)
%           XN    : data as numeric array, normalised length (coarse)
%           XFd   : functional data object equivalent of X
%           Y     : associated variable (e.g. class, outcome)
%           setup : initialised setup structure
%
% ************************************************************************


function [ X, XN, XFd, Y, setup ] = initializeData( source, nCodes, ...
                                            nPts, nPtsFine )

% get data
setup.source = source;

switch source
    case 'Synthetic'
        setup.synth.ratio = [ 4 8 16];
        setup.synth.sharedLevel = 3;
        setup.synth.mu = [1 4 8];
        setup.synth.sigma = [1 6 1];
        setup.synth.eta = 0.1;
        setup.synth.warpLevel = 2;
        setup.synth.tau = 50;

        setup.tSpan = linspace( 0, 1024, 33 );
        setup.tFine = linspace( 0, 1000, 21 );
        setup.synth.tSpan = setup.tSpan;

        N = 200;
        classSizes = [ N N N ];

        XRaw = genSyntheticData( classSizes, 1, setup.synth );
        Y = [ repelem(1,N) repelem(2,N) repelem(3,N) ]';

        setup.nDraw = 1;
        setup.cLabels = categorical( 0:length(classSizes) );
        setup.cDim = length( setup.cLabels );
        setup.nChannels = 1;

    case 'JumpVGRF'
        [ XRaw, Y ] = getJumpGRFData;
        Y = Y + 1;
        XLen = cellfun( @length, XRaw );
        maxLen = max( XLen );
        padLen = min( 1500, maxLen );
        padLoc = 'start';

        XRaw = padData( XRaw, padLen, 1, 'start' );

        tStart = -padLen+1;
        tEnd = 0;
        padValue = 1;
        normalization = 'PAD';

        setup.fda.basisOrder = 4;
        setup.fda.penaltyOrder = 2;
        setup.fda.lambda = 1E5; % 1E2
        setup.fda.nBasis = fix(padLen/10)+setup.fda.penaltyOrder;

        setup.nDraw = 1;
        setup.cLabels = categorical( 0:2 );
        setup.cDim = length( setup.cLabels );
        setup.nChannels = 1;

    case 'MSFT'
        [ XRaw, Y ] = getMFTData;
        XLen = cellfun( @length, XRaw );
        maxLen = max( XLen );
        padLen = maxLen;
        padLoc = 'both';

        XRaw = padData( XRaw, padLen, 'same', 'start' ); 

        tStart = 1;
        tEnd = padLen;
        normalization = 'LTN';

        setup.fda.basisOrder = 4;
        setup.fda.penaltyOrder = 2;
        setup.fda.lambda = 1E-2;
        setup.fda.nBasis = fix(padLen/4)+setup.fda.penaltyOrder;

        setup.nDraw = 1;
        setup.cLabels = categorical( 0:max(Y) );
        setup.cDim = length( setup.cLabels );
        setup.nChannels = 3;

    otherwise
        error('Unrecognised data source.');
end

% functional data analysis parameters
setup.fda.basisFd = create_bspline_basis( ...
                        [ tStart, tEnd ], ...
                          setup.fda.nBasis, setup.fda.basisOrder);
setup.fda.fdPar = fdPar( setup.fda.basisFd, ...
                         setup.fda.penaltyOrder, ...
                         setup.fda.lambda );

setup.fda.tSpan = linspace( tStart, tEnd, nPts );
setup.fda.tFine = linspace( tStart, tEnd, nPtsFine );

% smooth the data
XFd = smooth_basis( linspace( tStart, tEnd, padLen ), ...
                    XRaw, setup.fda.fdPar );

% create cell array of time series with variable lengths
% assumes that padding is at the start
XFine = eval_fd( setup.fda.tFine, XFd );
nObs = size( XFine,2);
X = cell( nObs, 1 );
for i = 1:nObs
    XLen(i) = ceil( nPtsFine*XLen(i)/maxLen );
    X{i} = squeeze(XFine( nPtsFine-XLen(i)+1:end, i, : ));
end

% prepare normalized data of fixed length
switch normalization

    case 'LTN' % time normalization
        XN = timeNormalize( X, nPts );
    case 'PAD' % padding
        XN = padData( X, nPtsFine, padValue, padLoc );
        XN = timeNormalize( XN, nPts );
    otherwise
        error('Unrecognized normalization method.');

end

% data generation parameters
setup.zDim = nCodes;
setup.xDim = length( setup.fda.tSpan );
setup.xDimFine = length( setup.fda.tFine );

% data embedding parameters
setup.embedding = false;
setup.embed.nKernels = 1000;
setup.embed.nMetrics = 4;
setup.embed.sampleRatio = 0.05;
setup.embed.usePCA = true;
setup.embed.retainThreshold = 0.5; % percentage


end