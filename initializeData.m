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


function [ X, XFd, Y, setup ] = initializeData( source, nCodes, ...
                                                nPts, doPadding )

if nargin < 4
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

        setup.tSpan = linspace( -padLen+1, 0, padLen );
        setup.tFine = linspace( -padLen+1, 0, nPts );

        setup.nDraw = 1;
        setup.cLabels = categorical( 0:2 );
        setup.cDim = length( setup.cLabels );

    otherwise
        error('Unrecognised data source.');
end

% data generation parameters
setup.zDim = nCodes;
setup.xDim = length( setup.tFine );

% data embedding parameters
setup.embedding = true;
setup.nKernels = 100;
setup.candidateStart = 3; % *2+1
setup.nCandidates = 4;
setup.isInterdependent = false;

% functional data analysis parameters
setup.fda.basisOrder = 4;
setup.fda.penaltyOrder = 2;
setup.fda.lambda = 1E5; % 1E2
setup.fda.nBasis = setup.xDim+setup.fda.penaltyOrder;
setup.fda.basisFd = create_bspline_basis( ...
                        [ setup.tSpan(1), setup.tSpan(end) ], ...
                          setup.fda.nBasis, setup.fda.basisOrder);
setup.fda.fdPar = fdPar( setup.fda.basisFd, ...
                         setup.fda.penaltyOrder, ...
                         setup.fda.lambda );
setup.fda.tSpan = setup.tFine;

% smooth the data
XFd = smooth_basis( setup.tSpan, XRaw, setup.fda.fdPar );
% re-sample it
X = eval_fd( setup.tFine, XFd );

if ~doPadding
    % re-package the data into a cell array with variable lengths
    nObs = size(X,2);
    XCell = cell( nObs, 1 );
    for i = 1:nObs
        adjLen = ceil( nPts*XLen(i)/maxLen );
        XCell{i} = X( nPts-adjLen+1:end, i );
    end
    X = XCell;
end


end