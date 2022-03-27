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
switch source
    case 'Synthetic'
        [XRaw, Y, XLen, setup] = initSyntheticData;

    case 'JumpVGRF'
        [XRaw, Y, XLen, setup] = initJumpVGRFData;

    case 'MSFT'
        [XRaw, Y, XLen, setup] = initMSFTData;

    otherwise
        error('Unrecognised data source.');
end
setup.source = source;


% smooth the raw data
tSpanRaw = 1:setup.maxLen;
tSpanResampled = linspace( 1, setup.maxLen, ...
                           fix(setup.maxLen/setup.resample)+1 );

rawBasisFd = create_bspline_basis( [1 setup.maxLen], ...
                                   setup.fda.nBasis, ...
                                   setup.fda.basisOrder );
rawFdPar = fdPar( rawBasisFd, ...
                  setup.fda.penaltyOrder, ...
                  setup.fda.lambda );

% smooth
XFd = smooth_basis( tSpanRaw, XRaw, rawFdPar );
% resample
XRaw = eval_fd( tSpanResampled, XFd );
% adjust lengths
XLen = ceil( size( XRaw, 1 )*XLen/setup.maxLen );
setup.maxLen = size( XRaw, 1 );


% re-create cell array of time series after smoothing
nObs = size( XRaw, 2 );
X = cell( nObs, 1 );
switch setup.padLoc
    case 'left'
        for i = 1:nObs
            X{i} = squeeze(XRaw( setup.maxLen-XLen(i)+1:end, i, : ));
        end
    case 'right'
        for i = 1:nObs
            X{i} = squeeze(XRaw( 1:XLen(i), i, : ));
        end
    case 'both'
        for i = 1:nObs
            adjLen = fix( (setup.maxLen-XLen(i))/2 );
            X{i} = squeeze(XRaw( adjLen+1:end-adjLen, i, : ));
        end
end


% prepare normalized data of fixed length
switch setup.normalization

    case 'LTN' % time normalization
        XN = timeNormalize( X, nPts );
    case 'PAD' % padding
        XN = padData( X, setup.maxLen, setup.padValue, setup.padLoc );
        XN = timeNormalize( XN, nPts );
    otherwise
        error('Unrecognized normalization method.');

end


% functional data analysis parameters
setup.fda.basisFd = create_bspline_basis( ...
                        [ setup.tStart, setup.tEnd ], ...
                          setup.fda.nBasis, setup.fda.basisOrder);
setup.fda.fdPar = fdPar( setup.fda.basisFd, ...
                         setup.fda.penaltyOrder, ...
                         setup.fda.lambda );

setup.fda.tSpan = linspace( setup.tStart, setup.tEnd, nPts );
setup.fda.tFine = linspace( setup.tStart, setup.tEnd, nPtsFine );


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



function [XRaw, Y, XLen, setup ] = initSyntheticData

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

    setup.maxLen = 33;
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



function [XRaw, Y, XLen, setup ] = initJumpVGRFData

    [ XRaw, Y ] = getJumpGRFData;
    Y = Y + 1;

    % get raw lengths
    XLen = cellfun( @length, XRaw );
    setup.maxLen = max( XLen );
    setup.resample = 10;

    % setup padding
    XLenLimit = 1500;
    setup.normalization = 'PAD';
    setup.padLen = min( setup.maxLen, XLenLimit );
    setup.padLoc = 'left';
    setup.padValue = 1;

    % initially pad the series for smoothing
    XRaw = padData( XRaw, setup.padLen, setup.padValue, setup.padLoc );

    % revise the lengths taking into the limit
    XLen = min( XLen, XLenLimit);
    setup.maxLen = max( XLen );

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


function [XRaw, Y, XLen, setup ] = initMSFTData

    [ XRaw, Y ] = getMFTData;

    XLen = cellfun( @length, XRaw );
    setup.maxLen = max( XLen );
    setup.resample = 1;

    setup.normalization = 'LTN';
    setup.padLen = maxLen;
    setup.padLoc = 'both';

    % initially pad the series for smoothing
    XRaw = padData( XRaw, padLen, 'same', setup.padLoc ); 

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