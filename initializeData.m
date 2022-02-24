% ************************************************************************
% Function: initializeData
%
% Initialise the data setup
%
% Parameters:
%           
% Outputs:
%           setup : initialised setup structure
%
% ************************************************************************


function [ XFine, X, XFd, Y, setup ] = initializeData( source, nCodes, ...
                                            nPts, nPtsFine, doPadding )

if nargin < 5
    doPadding = true;
end

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

    case 'JumpVGRF'
        [ XRaw, Y ] = getJumpGRFData;
        Y = Y + 1;
        XLen = cellfun( @length, XRaw );
        maxLen = max( XLen );
        if doPadding
            padLen = min( 1500, maxLen );
        else
            padLen = maxLen;
        end

        XRaw = padData( XRaw, padLen, 1 ); % always pad at this point

        tStart = -padLen+1;
        tEnd = 0;

        setup.nDraw = 1;
        setup.cLabels = categorical( 0:2 );
        setup.cDim = length( setup.cLabels );

    otherwise
        error('Unrecognised data source.');
end

% data embedding parameters
setup.embedding = true;
setup.embed.nKernels = 1000;
setup.embed.nMetrics = 4;
setup.embed.sampleRatio = 0.05;

% functional data analysis parameters
setup.fda.basisOrder = 4;
setup.fda.penaltyOrder = 2;
setup.fda.lambda = 1E5; % 1E2
setup.fda.nBasis = fix(padLen/10)+setup.fda.penaltyOrder;

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
% re-sample it
X = eval_fd( setup.fda.tSpan, XFd );
XFine = eval_fd( setup.fda.tFine, XFd );

if ~doPadding
    % re-package the data into a cell array with variable lengths
    nObs = size(X,2);
    XCell = cell( nObs, 1 );
    for i = 1:nObs
        adjLen = ceil( nPtsFine*XLen(i)/maxLen );
        XCell{i} = XFine( nPtsFine-adjLen+1:end, i );
    end
    X = XCell;
end

% data generation parameters
setup.zDim = nCodes;
setup.xDim = length( setup.fda.tSpan );
setup.xDimFine = length( setup.fda.tFine );


end