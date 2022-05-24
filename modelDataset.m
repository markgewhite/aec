classdef modelDataset
    % Class defining a dataset

    properties
        XInputRaw       % original, raw time series data
        XFd             % functional representation of the data
        XLen            % array recording the length of each series

        XInputDim       % number of X input dimensions
        XTargetDim      % normalized X output size (time series length)
        XChannels       % number of X input channels

        XDimLabels      % X's dlarray dimension labels

        Y               % outcome variable
        CDim            % number of categories
        YLabels         % Y labels
        nObs            % number of observations

        normalization   % structure for time-normalization 
        normalizedPts   % standardized number of points for normalization
        normalizeInput  % whether input should be time-normalized too
        matchingOutput  % whether input and target output must match

        padding         % structure specifying padding setup
        fda             % functional data analysis settings

        tSpan           % time span structure of vectors holding
                        %   .original = matching the raw data
                        %   .regular = regularly-spaced times
                        %   .input = input to the model (may be adaptive)
                        %   .target = for autoencoder reconstructions

        hasAdaptiveTimeSpan % flag whether to adjust timespan to complexity
        adaptiveLowerBound % lower limit on point density
        adaptiveUpperBound % upper limit on point density
        resampleRate    % downsampling rate
        overSmoothing   % additional factor for target roughness penalty 

        info            % dataset information (used for plotting)
    end

    properties (Dependent = true)
        XInput          % processed input data (variable length)       
        XTarget         % target output
        XInputRegular   % processed input with regularly spaced time span
    end


    methods

        function self = modelDataset( XInputRaw, Y, tSpan, dimLabels, args )
            % Create and preprocess the data.
            % The calling function will be a data loader or
            % a function partitioning the data.
            arguments
                XInputRaw                   cell
                Y
                tSpan                       double
                dimLabels                   char
                args.XTargetRaw             cell = []
                args.purpose                char ...
                    {mustBeMember( args.purpose, ...
                    {'Creation', 'ForSubset'} )} = 'Creation'
                args.normalization          char ...
                    {mustBeMember( args.normalization, ...
                    {'PAD', 'LTN'} )} = 'LTN'
                args.normalizedPts          double ...
                    {mustBeNumeric, mustBePositive, mustBeInteger} = 101
                args.normalizeInput         logical = false
                args.matchingOutput         logical = false
                args.padding                struct ...
                    {mustBeValidPadding}
                args.fda                    struct ...
                    {mustBeValidFdParams}
                args.tSpan                  double = []
                args.hasAdaptiveTimeSpan    logical = false
                args.adaptiveLowerBound     double = 0.05
                args.adaptiveUpperBound     double = 5
                args.resampleRate           double ...
                    {mustBeNumeric} = 1
                args.overSmoothing          double = 1E0
                args.datasetName            string
                args.timeLabel              string = "Time"
                args.channelLabels          string
                args.channelLimits          double
            end

            self.XInputRaw = XInputRaw;
            self.Y = Y;
            self.XDimLabels = dimLabels;
            
            if strcmp( args.purpose, 'ForSubset' )
                % don't do the setup
                return
            end

            % set properties
            self.normalization = args.normalization;
            self.normalizedPts = args.normalizedPts;
            self.normalizeInput = args.normalizeInput;
            self.matchingOutput = args.matchingOutput;
            self.padding = args.padding;
            self.fda = args.fda;

            self.tSpan.original = tSpan;
            self.hasAdaptiveTimeSpan = args.hasAdaptiveTimeSpan;
            self.adaptiveLowerBound = args.adaptiveLowerBound;
            self.adaptiveUpperBound = args.adaptiveUpperBound;
            self.resampleRate = args.resampleRate;
            self.fda.overSmoothing = args.overSmoothing;

            self.info.datasetName = args.datasetName;
            self.info.channelLabels = args.channelLabels;
            self.info.timeLabel = args.timeLabel;
            self.info.channelLimits = args.channelLimits;

            % get immediately available dimensions
            self.nObs = length( XInputRaw );
            self.XChannels = size( XInputRaw{1}, 2 );

            % create smooth functions for the data
            [self.XFd, self.XLen] = smoothRawData( XInputRaw, ...
                                                   self.padding, ...
                                                   self.tSpan.original, ...
                                                   self.fda  );

            % re-sampling at the given rate
            self.tSpan.regular = linspace( ...
                    self.tSpan.original(1), ...
                    self.tSpan.original(end), ...
                    fix( self.padding.length/self.resampleRate ) );  

            % adaptive resampling, as required
            if self.hasAdaptiveTimeSpan && isempty( args.tSpan )

                self.tSpan.input = calcAdaptiveTimeSpan( ...
                    self.XFd, ...
                    self.tSpan.regular, ...
                    self.adaptiveLowerBound, ...
                    self.adaptiveUpperBound );

            elseif isempty( args.tSpan )
                self.tSpan.input = self.tSpan.regular;
            else
                self.tSpan.input = args.tSpan;
            end

            if self.matchingOutput
                % ensure the input and target time spans match
                % this usually applies for PCA
                self.tSpan.target = self.tSpan.regular;

            else
                % adjust the time span in proportion to number of points
                tSpan0 = 1:length( self.tSpan.input );
                tSpan1 = linspace( 1, length(self.tSpan.input), ...
                                   self.normalizedPts );

                self.tSpan.target= interp1( ...
                            tSpan0, self.tSpan.input, tSpan1 );
            end

            self.XInputDim = length( self.tSpan.input );
            self.XTargetDim = self.normalizedPts;

            % set the FD parameters for regular spacing
            self.fda.fdParamsRegular = setFDAParameters( ...
                                            self.tSpan.regular, ...
                                            self.fda.basisOrder, ...
                                            self.fda.penaltyOrder, ...
                                            self.fda.lambda );

            % set the FD parameters for adaptive spacing
            self.fda.fdParamsInput = setFDAParameters( ...
                                            self.tSpan.input, ...
                                            self.fda.basisOrder, ...
                                            self.fda.penaltyOrder, ...
                                            self.fda.lambda );

            % set the FD parameters for adaptive spacing
            % with a higher level of smoothing than the input
            lambda = self.fda.lambda*self.fda.overSmoothing;
            self.fda.fdParamsTarget = setFDAParameters( ...
                                    self.tSpan.target, ...
                                    self.fda.basisOrder, ...
                                    self.fda.penaltyOrder, ...
                                    lambda );

            % assign category labels
            self.YLabels = categorical( unique(self.Y) );
            self.CDim = length( self.YLabels );          

        end


        function thisSubset = partition( self, idx )
            % Create the subset of this modelDataset
            % using the indices specified
            arguments
                self        modelDataset
                idx         logical 
            end

            subXRaw = split( self.XInputRaw, idx );
            subY = self.Y( idx );
            subTSpan = self.tSpan.original;
            subLabels = self.XDimLabels;

            thisSubset = modelDataset( subXRaw, subY, subTSpan, subLabels, ...
                                       purpose='ForSubset' );

            thisSubset.XFd = splitFd( self.XFd, idx );
            thisSubset.XLen = self.XLen( idx );

            thisSubset.normalization = self.normalization;
            thisSubset.normalizedPts = self.normalizedPts;
            thisSubset.normalizeInput = self.normalizeInput;

            thisSubset.padding = self.padding;
            thisSubset.fda = self.fda;
            thisSubset.resampleRate = self.resampleRate;

            thisSubset.XInputDim = self.XInputDim;
            thisSubset.XTargetDim = self.XTargetDim;
            thisSubset.XChannels = self.XChannels;
            thisSubset.XDimLabels = self.XDimLabels;

            thisSubset.CDim = self.CDim;
            thisSubset.YLabels = self.YLabels;
            thisSubset.nObs = sum( idx );

            thisSubset.tSpan = self.tSpan;

            thisSubset.matchingOutput = self.matchingOutput;
            thisSubset.hasAdaptiveTimeSpan = self.hasAdaptiveTimeSpan;
            thisSubset.adaptiveLowerBound = self.adaptiveLowerBound;
            thisSubset.adaptiveUpperBound = self.adaptiveUpperBound;

            thisSubset.info = self.info;

        end


        function isFixed = isFixedLength( self )
            % return whether data is time-normalized
            arguments
                self    modelDataset
            end

            isFixed = self.normalizeInput;
            
        end



        function [ X, Y ] = getDLInput( self, arg )
            % Convert X and Y into dl arrays
            arguments
                self            modelDataset
                arg.dlarray     logical = true
            end
            
            X = padData( self.XInput, 0, self.padding.value, ...
                         Longest = true, ...
                         Same = self.padding.same, ...
                         Location = self.padding.location );

            X = dlarray( X, self.XDimLabels );

            if arg.dlarray
                Y = dlarray( self.Y, 'CB' );
            else
                Y = self.Y;
            end

        end


        function validateSmoothing( self )
            % Re-run smoothing with maximum flexibility
            arguments
                self        modelDataset
            end

            % pad the raw series for smoothing
            X = padData( self.XInputRaw, ...
                         self.padding.length, ...
                         self.padding.value, ...
                         Same = self.padding.same, ...
                         Location = self.padding.location, ...
                         Anchoring = self.padding.anchoring );

            % create a time span with maximum detail
            thisTSpan = linspace( self.tSpan.original(1),...
                              self.tSpan.original(end), ...
                              size( X, 1 ) );

            % create a new basis with maximum number of functions
            basis = create_bspline_basis( [thisTSpan(1) thisTSpan(end)], ...
                                          size( X, 1 ), ...
                                          self.fda.basisOrder);
            
            % Find minimum GCV value of lambda
            % search for the best value for lambda, the roughness penalty
            logLambda   = -12:1:6;
            gcvSave = zeros( length(logLambda), self.XChannels );
            dfSave  = zeros( length(logLambda), 1 );
            
            for i = 1:length(logLambda)
                
                % set smoothing parameters
                XFdPari = fdPar( basis, ...
                                 self.fda.penaltyOrder, ...
                                 10^logLambda(i) );
                
                % perform smoothing
                [~, dfi, gcvi] = smooth_basis( thisTSpan, X, XFdPari );
                
                % determine mean GCV and degrees of freedom
                gcvSave(i,:) = sqrt( sum( gcvi, 2 )/self.nObs ); 
                dfSave(i)  = dfi;
                
            end
            
            %  plot the results for GCV and DF
            figure;
            
            plot( logLambda, log10(gcvSave), 'k-o' );
            ylabel('\fontsize{13} log_{10}( GCV )');
            hold on;
            
            yyaxis right;
            plot( logLambda, log10(dfSave), 'r-o' );
            ylabel('\fontsize{13} log_{10}( DF )');
            
            xlabel('\fontsize{13} log_{10}(\lambda)');
            
        end


        function X = get.XInputRegular( self )
            % Generate the regularly-spaced input from XFd
            arguments
                self    modelDataset
            end

            XCell = processX( self.XFd, ...
                               self.XLen, ...
                               self.tSpan.original, ...
                               self.tSpan.regular, ...
                               self.padding, ...
                               true, ...
                               length(self.tSpan.regular), ...
                               self.normalization );

            X = reshape( cell2mat( XCell ), [], self.nObs, self.XChannels );

        end


        function X = get.XInput( self )
            % Generate the adaptively-spaced input from XFd
            arguments
                self    modelDataset
            end

            X = processX(  self.XFd, ...
                           self.XLen, ...
                           self.tSpan.original, ...
                           self.tSpan.input, ...
                           self.padding, ...
                           self.normalizeInput, ...
                           length(self.tSpan.input), ...
                           self.normalization );

        end
        

        function X = get.XTarget( self )
            % Generate the adaptively-spaced input from XFd
            % producing an array of fixed length
            arguments
                self    modelDataset
            end

            if self.matchingOutput
                numPts = self.XInputDim;
            else
                numPts = self.normalizedPts;
            end

            XCell = processX(  self.XFd, ...
                               self.XLen, ...
                               self.tSpan.original, ...
                               self.tSpan.target, ...
                               self.padding, ...
                               true, ...
                               numPts, ...
                               self.normalization );

            X = timeNormalize( XCell, numPts );

        end
        
    end


    methods (Static)

        function mbq = getMiniBatchQueue( self, batchSize, args )
            % Create a minibatch queue
            arguments
                self                modelDataset
                batchSize           double ...
                    {mustBeInteger, mustBePositive}
                args.partialBatch   char ...
                    {mustBeMember( args.partialBatch, ...
                    {'return', 'discard'} )} = 'discard'
            end

            ds = createDatastore( self.XInput, self.XTarget, self.Y );

            % setup the minibatch preprocessing function
            preproc = @( X, XN, Y ) preprocMiniBatch( X, XN, Y, ...
                          self.padding.value, ...
                          self.padding.location );

            mbq = minibatchqueue(  ds,...
                  MiniBatchSize = batchSize, ...
                  PartialMiniBatch = args.partialBatch, ...
                  MiniBatchFcn = preproc, ...
                  MiniBatchFormat = {self.XDimLabels, self.XDimLabels, 'CB'} );

        end


    end

end


function [XFd, XLen] = smoothRawData( XCell, padding, tSpan, fda )

    % find the series lengths (capped at padLen)
    XLen = min( cellfun( @length, XCell ), padding.length );

    % pad the series for smoothing
    X = padData( XCell, padding.length, padding.value, ...
                 Same = padding.same, ...
                 Location = padding.location, ...
                 Anchoring = padding.anchoring );
    
    % setup the smoothing parameters
    fdParams = setFDAParameters( tSpan, ...
                                 fda.basisOrder, fda.penaltyOrder, ...
                                 fda.lambda );

    % create the smooth functions
    XFd = smooth_basis( tSpan, X, fdParams );

end



function fdParams = setFDAParameters( tSpan, ...
                                      basisOrder, penaltyOrder, lambda )
    % Setup the FDA parameters object
    arguments
        tSpan           double
        basisOrder      double {mustBeInteger, mustBePositive}
        penaltyOrder    double {mustBeInteger, mustBePositive}
        lambda          double
    end

    % create a basis for smoothing with a knot at each point
    % with one function per knot
    nBasis = length( tSpan ) + penaltyOrder;

    basisFd = create_bspline_basis( [tSpan(1) tSpan(end)], ...
                                    nBasis, ...
                                    basisOrder );

    % setup the smoothing parameters
    fdParams = fdPar( basisFd, penaltyOrder, lambda );

end


function [ X, XDim ] = processX( ...
                                XFd, XLen, tSpan, tSpanNew, pad, ...
                                normalize, normalizedPts, normalization )

    % evaluate the input function at these points
    XEval = eval_fd( tSpanNew, XFd );
   
    % adjust lengths for non-linear re-sampling
    XLenNew = adjustXLengths( XLen, tSpan, tSpanNew, pad.location );
    
    % re-scale for resampled length
    pad.length = max( XLenNew );
    
    % recreate the cell time series
    XCell = extractXSeries( XEval, XLenNew, pad.length, pad.location );

    if normalize
        % use time-normalization method to set a fixed length
        XNorm = normalizeXSeries( XCell, normalizedPts, ...
                                        normalization, ...
                                        pad );
        XDim = size( XNorm, 1);
        if size( XNorm, 3 ) > 1
            X = num2cell( permute( XNorm, [2 1 3]), [2 3] );
            X = cellfun( @squeeze, X , 'UniformOutput', false);
        else
            X = num2cell( permute( XNorm, [2 1]), 2 );
            X = cellfun( @transpose, X , 'UniformOutput', false);
        end
        
    else
        % has variable length input
        X = XCell;
        XDim = 1;
    end

end


function tSpanAdaptive = calcAdaptiveTimeSpan( XFd, tSpan, ...
                                               lowerBound, upperBound ) 

    % evaluate the mean XFd curvature (2nd derivative)
    D1XEval = squeeze(mean( abs(eval_fd( tSpan, XFd, 1 )), 2));
    D2XEval = squeeze(mean( abs(eval_fd( tSpan, XFd, 2 )), 2));

    D1XEval = min( max( D1XEval./mean(D1XEval), lowerBound ), upperBound );
    D2XEval = min( max( D2XEval./mean(D2XEval), lowerBound ), upperBound );
    
    DXEvalComb = sum( D1XEval + D2XEval, 2 );

    % cumulatively sum the absolute inverse curvatures
    % inserting zero at the begining to ensure first point will be at 0
    D2XInt = cumsum( [0; 1./DXEvalComb] );
    D2XInt = D2XInt./max(D2XInt);

    % normalize to the tSpan
    tSpanAdaptive = tSpan(1) + D2XInt*(tSpan(end)-tSpan(1));

    % reinterpolate to remove the extra point
    nPts = length( tSpan );
    tSpanAdaptive = interp1( 1:nPts+1, ...
                             tSpanAdaptive, ...
                             linspace(1, nPts+1, nPts) );

end


function XLen = adjustXLengths( XLen, tSpan, tSpanAdaptive, padding )

    for i = 1:length(XLen)
        switch padding

            case 'Left'
                tEnd = tSpan( length(tSpan)-XLen(i)+1 );
                XLen(i) = length(tSpanAdaptive) - find( tEnd <= tSpanAdaptive, 1 ) + 1;

            case {'Right', 'Both'}
                tEnd = tSpan( XLen(i) );
                XLen(i) = find( tEnd <= tSpanAdaptive, 1 );

        end
    end

end


function XCell = extractXSeries( X, XLen, maxLen, padLoc )

    nObs = length( XLen );
    XCell = cell( nObs, 1 );
    switch padLoc
        case 'Left'
            for i = 1:nObs
                XCell{i} = squeeze(X( maxLen-XLen(i)+1:end, i, : ));
            end
        case 'Right'
            for i = 1:nObs
                XCell{i} = squeeze(X( 1:XLen(i), i, : ));
            end
        case 'Both'
            for i = 1:nObs
                adjLen = fix( (maxLen-XLen(i))/2 );
                XCell{i} = squeeze(X( adjLen+1:end-adjLen, i, : ));
            end
    end

end


function XN = normalizeXSeries( X, nPts, type, pad )

    switch type
    
        case 'LTN' % time normalization
            XN = timeNormalize( X, nPts );

        case 'PAD' % padding
            XN = padData( X, pad.length, pad.value, ...
                             Same = pad.same, ...
                             Location = pad.location, ...
                             Anchoring = pad.anchoring );
            XN = timeNormalize( XN, nPts );
    
    end

end


function XS = split( X, indices )

    if iscell( X )
        XS = X( indices );
    else
        XS = X( :, indices, : );
    end

end


function XSFd = splitFd( XFd, indices )

    coeff = getcoef( XFd );
    coeff( :, ~indices, : ) = [];
    XSFd = putcoef( XFd, coeff );

end


function dsFull = createDatastore( X, XN, Y )

    % create the datastore for the input X
         
    % sort them in ascending order of length
    %XLen = cellfun( @length, X );
    %[ ~, orderIdx ] = sort( XLen, 'descend' );

    %X = X( orderIdx );
    dsX = arrayDatastore( X, 'IterationDimension', 1, ...
                             'OutputType', 'same' );
       
    % create the datastore for the time-normalised output X
    dsXN = arrayDatastore( XN, 'IterationDimension', 2 );
    
    % create the datastore for the labels/outcomes
    dsY = arrayDatastore( Y, 'IterationDimension', 1 );   
    
    % combine them
    dsFull = combine( dsX, dsXN, dsY );
               
end


function [ X, XN, Y ] = preprocMiniBatch( XCell, XNCell, YCell, ...
                                          padValue, padLoc )
    % Preprocess a sequence batch for training

    X = padData( XCell, 0, padValue, Longest = true, Location = padLoc  );
    
    if ~isempty( XNCell )
        XN = cat( 2, XNCell{:} );   
    else
        XN = [];
    end
    
    if ~isempty( YCell )
        Y = cat( 2, YCell{:} );
    else
        Y = [];
    end


end
