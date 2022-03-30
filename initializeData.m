% ************************************************************************
% Function: initializeData
%
% Initialise the data setup
%
% Parameters:
%           source : char array identifier of the data source
%           nCodes : number of latent codes (to be stored)
%           nPts   : number of resampled points for time normalisation
%           
% Outputs:
%           X     : data as cell array, variable length (fine)
%           XN    : data as numeric array, normalised length (coarse)
%           XFd   : functional data object equivalent of X
%           Y     : associated variable (e.g. class, outcome)
%           setup : initialised setup structure
%
% ************************************************************************


function [ X, XN, XFd, Y, setup ] = initializeData( source, nCodes, nPts, outcome )

% get data
switch source
    case 'Synthetic'
        [XRaw, Y, setup] = initSyntheticData;

    case 'JumpVGRF'
        [XRaw, Y, setup] = initJumpVGRFData;

    case 'JumpACC'
        [XRaw, Y, setup] = initJumpACCData( outcome );

    case 'MSFT'
        [XRaw, Y, setup] = initMSFTData;

    otherwise
        error('Unrecognised data source.');
end
setup.source = source;

% create smooth functions for the data
[XFd, XLen] = smoothRawData( XRaw, setup );

% re-create cell array of time series after smoothing
% resampling, as required
tSpanResampled = linspace( 1, setup.padLen, ...
                              fix(setup.padLen/setup.resample)+1 );

XEval = eval_fd( tSpanResampled, XFd );

if setup.derivative
    % include the first derivative as further channels
    DXEval = eval_fd( tSpanResampled, XFd, 1 );
    XEval = cat( 3, XEval, DXEval );
end

% adjust lengths
XLen = ceil( size( XEval, 1 )*XLen/setup.padLen );
setup.padLen = max( XLen );

% recreate the cell time series
X = extractXSeries( XEval, XLen, setup.padLen, setup.padLoc );

% prepare normalized data of fixed length
XN = normalizeXSeries( X, nPts, setup );


% functional data analysis parameters
setup.fda.basisFd = create_bspline_basis( ...
                        [ setup.tStart, setup.tEnd ], ...
                          setup.fda.nBasis, setup.fda.basisOrder);
setup.fda.fdPar = fdPar( setup.fda.basisFd, ...
                         setup.fda.penaltyOrder, ...
                         setup.fda.lambda );

setup.fda.tSpan = linspace( setup.tStart, setup.tEnd, nPts );


% data generation parameters
setup.zDim = nCodes;
setup.xDim = length( setup.fda.tSpan );

% data embedding parameters
setup.embedding = false;
setup.embed.nKernels = 1000;
setup.embed.nMetrics = 4;
setup.embed.sampleRatio = 0.05;
setup.embed.usePCA = true;
setup.embed.retainThreshold = 0.5; % percentage


end


function [XFd, XLen] = smoothRawData( X, setup )

    % find the series lengths (capped at padLen)
    XLen = min( cellfun( @length, X ), setup.padLen );

    % pad the series for smoothing
    X = padData( X, setup.padLen, setup.padValue, setup.padLoc );
    
    % create a basis for smoothing with a knot at each point
    basisFd = create_bspline_basis( [1 setup.padLen], ...
                                       setup.fda.nBasis, ...
                                       setup.fda.basisOrder );
    % setup the smoothing parameters
    fdParams = fdPar( basisFd, ...
                      setup.fda.penaltyOrder, ...
                      setup.fda.lambda );

    % create the smooth functions
    XFd = smooth_basis( 1:setup.padLen, X, fdParams );

end


function XCell = extractXSeries( X, XLen, maxLen, padLoc )

    nObs = length( XLen );
    XCell = cell( nObs, 1 );
    switch padLoc
        case 'left'
            for i = 1:nObs
                XCell{i} = squeeze(X( maxLen-XLen(i)+1:end, i, : ));
            end
        case 'right'
            for i = 1:nObs
                XCell{i} = squeeze(X( 1:XLen(i), i, : ));
            end
        case 'both'
            for i = 1:nObs
                adjLen = fix( (maxLen-XLen(i))/2 );
                XCell{i} = squeeze(X( adjLen+1:end-adjLen, i, : ));
            end
    end

end


function XN = normalizeXSeries( X, nPts, setup )

    switch setup.normalization
    
        case 'LTN' % time normalization
            XN = timeNormalize( X, nPts );
        case 'PAD' % padding
            XN = padData( X, setup.padLen, setup.padValue, setup.padLoc );
            XN = timeNormalize( XN, nPts );
        otherwise
            error('Unrecognized normalization method.');
    
    end

end



function [XRaw, Y, setup ] = initSyntheticData

    N = 200;
    classSizes = [ N N N ];

    setup.synth.ratio = [ 4 8 16];
    setup.synth.sharedLevel = 3;
    setup.synth.mu = [1 4 8];
    setup.synth.sigma = [1 6 1];
    setup.synth.eta = 0.1;
    setup.synth.warpLevel = 2;
    setup.synth.tau = 50;

    XRaw = genSyntheticData( classSizes, 1, setup.synth );
    Y = [ repelem(1,N) repelem(2,N) repelem(3,N) ]';

    setup.padLen = 33;
    XLen = ones( 3*N, 1 )*maxLen;
    setup.resample = 1;

    setup.tSpan = linspace( 0, 1024, 33 );
    setup.tFine = linspace( 0, 1000, 21 );
    setup.synth.tSpan = setup.tSpan;

    setup.nDraw = 1;
    setup.cLabels = categorical( 0:length(classSizes) );
    setup.cDim = length( setup.cLabels );
    setup.nChannels = 1;

end



function [XRaw, Y, setup ] = initJumpVGRFData

    [ XRaw, Y ] = getJumpGRFData;
    Y = Y + 1;

    setup.resample = 10;
    setup.derivative = false;

    % setup padding
    setup.normalization = 'PAD';
    setup.padLen = 1500;
    setup.padLoc = 'left';
    setup.padValue = 1;

    setup.tStart = -setup.padLen+1;
    setup.tEnd = 0;

    setup.fda.basisOrder = 4;
    setup.fda.penaltyOrder = 2;
    setup.fda.lambda = 1E5; % 1E2
    setup.fda.nBasis = fix( setup.padLen/10 )+setup.fda.penaltyOrder;

    setup.nDraw = 1;
    setup.cLabels = categorical( 0:2 );
    setup.cDim = length( setup.cLabels );
    setup.nChannels = 1;


end


function [XRaw, Y, setup ] = initJumpACCData( outcome )

    [ XRaw, Y ] = getJumpACCData( outcome );
    if strcmpi( outcome, 'JumpType' )
        Y = Y + 1;
    end

    setup.derivative = false;
    setup.resample = 1;

    % setup padding
    setup.normalization = 'PAD';
    setup.padLen = 5000;
    setup.padLoc = 'both';
    setup.padValue = 'same';

    setup.tStart = -setup.padLen+1;
    setup.tEnd = 0;

    setup.fda.basisOrder = 4;
    setup.fda.penaltyOrder = 2;
    setup.fda.lambda = 1E2; % 1E2
    setup.fda.nBasis = fix( setup.padLen/10 )+setup.fda.penaltyOrder;

    setup.nDraw = 1;
    setup.cLabels = categorical( 0:2 );
    setup.cDim = length( setup.cLabels );
    setup.nChannels = 1;


end

function [XRaw, Y, setup ] = initMSFTData

    [ XRaw, Y ] = getMFTData;

    setup.resample = 1;
    setup.derivative = false;

    setup.normalization = 'LTN';
    setup.padLen = max( cellfun(@length, XRaw) );
    setup.padLoc = 'both';

    setup.tStart = 1;
    setup.tEnd = setup.padLen;

    setup.fda.basisOrder = 4;
    setup.fda.penaltyOrder = 2;
    setup.fda.lambda = 1E-2;
    setup.fda.nBasis = fix(padLen/4)+setup.fda.penaltyOrder;

    setup.nDraw = 1;
    setup.cLabels = categorical( 0:max(Y) );
    setup.cDim = length( setup.cLabels );
    setup.nChannels = 3;

end