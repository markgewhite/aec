classdef modelDataset
    % Class defining a dataset

    properties
        XInputRaw       % original, raw time series data
        XInput          % processed input data (variable length)
        XInputRegular   % processed input with regularly spaced time span
        XTarget         % target output

        XInputDim       % number of X input dimensions
        XTargetDim      % normalized X output size (time series length)
        XChannels       % number of X input channels
        XInputLen       % lengths of the input series

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
        adaptiveTimeSpan % flag whether to adjust timespan to complexity
        adaptiveLowerBound % lower limit on point density
        adaptiveUpperBound % upper limit on point density
        resampleRate    % downsampling rate

        info            % dataset information (used for plotting)
    end


    methods

        function self = modelDataset( XInputRaw, Y, dimLabels, args )
            % Create and preprocess the data.
            % The calling function will be a data loader or
            % a function partitioning the data.
            arguments
                XInputRaw               cell
                Y
                dimLabels               char
                args.XTargetRaw         cell = []
                args.purpose            char ...
                    {mustBeMember( args.purpose, ...
                    {'Creation', 'ForSubset'} )} = 'Creation'
                args.normalization      char ...
                    {mustBeMember( args.normalization, ...
                    {'PAD', 'LTN'} )} = 'LTN'
                args.normalizedPts      double ...
                    {mustBeNumeric, mustBePositive, mustBeInteger} = 101
                args.normalizeInput     logical = false
                args.matchingOutput     logical = false
                args.padding            struct ...
                    {mustBeValidPadding}
                args.fda                struct ...
                    {mustBeValidFdParams}
                args.adaptiveTimeSpan   logical = false
                args.adaptiveLowerBound double = 1E-8
                args.adaptiveUpperBound double = 1E-1
                args.resampleRate       double ...
                    {mustBeNumeric} = 1
                args.overSmoothing      double = 1E2
                args.datasetName        string
                args.timeLabel          string = "Time"
                args.channelLabels      string
                args.channelLimits      double
            end

            self.XInputRaw = XInputRaw;
            self.Y = Y;
            self.XDimLabels = dimLabels;
            
            if strcmp( args.purpose, 'ForSubset' )
                % don't do the setup
                return
            end

            self.normalization = args.normalization;
            self.normalizedPts = args.normalizedPts;
            self.normalizeInput = args.normalizeInput;
            self.matchingOutput = args.matchingOutput;
            self.padding = args.padding;
            self.fda = args.fda;
            self.adaptiveTimeSpan = args.adaptiveTimeSpan;
            self.adaptiveLowerBound = args.adaptiveLowerBound;
            self.adaptiveUpperBound = args.adaptiveUpperBound;
            self.resampleRate = args.resampleRate;
            self.fda.overSmoothing = args.overSmoothing;

            self.info.datasetName = args.datasetName;
            self.info.channelLabels = args.channelLabels;
            self.info.timeLabel = args.timeLabel;
            self.info.channelLimits = args.channelLimits;

            % prepare the input data
            self = processXSeries( self, XInputRaw );

            % prepare the target data
            self = prepareXTarget( self, args.XTargetRaw );

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
            subLabels = self.XDimLabels;

            thisSubset = modelDataset( subXRaw, subY, subLabels, ...
                                       purpose='ForSubset' );

            thisSubset.normalization = self.normalization;
            thisSubset.normalizedPts = self.normalizedPts;
            thisSubset.normalizeInput = self.normalizeInput;

            thisSubset.padding = self.padding;
            thisSubset.fda = self.fda;
            thisSubset.resampleRate = self.resampleRate;

            thisSubset.XInput = split( self.XInput, idx );
            thisSubset.XInputRegular = split( self.XInputRegular, idx );
            thisSubset.XTarget = split( self.XTarget, idx );
            thisSubset.XInputLen = self.XInputLen( idx );

            thisSubset.XInputDim = self.XInputDim;
            thisSubset.XTargetDim = self.XTargetDim;
            thisSubset.XChannels = self.XChannels;
            thisSubset.XDimLabels = self.XDimLabels;

            thisSubset.CDim = self.CDim;
            thisSubset.YLabels = self.YLabels;
            thisSubset.nObs = sum( idx );

            thisSubset.info = self.info;

        end


        function isFixed = isFixedLength( self )
            % return whether data is time-normalized
            arguments
                self    modelDataset
            end

            isFixed = self.normalizeInput;
            
        end



        function [ X, Y ] = getInput( self, arg )
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
            tSpan = linspace( self.fda.tSpan(1),...
                              self.fda.tSpan(end), ...
                              size( X, 1 ) );

            % create a new basis with maximum number of functions
            basis = create_bspline_basis( [tSpan(1) tSpan(end)], ...
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
                [~, dfi, gcvi] = smooth_basis( tSpan, X, XFdPari );
                
                % determine mean GCV and degrees of freedom
                gcvSave(i,:) = sqrt( sum( gcvi )/self.nObs ); 
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
        
        
    end


    methods (Access = private)    
        

        function self = processXSeries( self, XInputRaw )
            % Prepare the input data by smoothing, re-sampling
            % adding the 1st derivative as a separate channel
            % XInputRaw and XInput are cell arrays
            arguments
                self        modelDataset
                XInputRaw   cell
            end

            % get immediately available dimensions
            self.nObs = length( XInputRaw );
            self.XChannels = size( XInputRaw{1}, 2 );

            % create smooth functions for the data
            [XFd, XLen] = smoothRawData( XInputRaw, self.padding, self.fda  );

            % re-sampling at the given rate
            self.fda.tSpanRegular = linspace( ...
                    self.fda.tSpan(1), self.fda.tSpan(end), ...
                    fix( self.padding.length/self.resampleRate ) );  

            % adaptive resampling, as required
            if self.adaptiveTimeSpan
                self.fda.tSpanInput = calcAdaptiveTimeSpan( ...
                    XFd, self.fda.tSpan, ...
                    self.resampleRate, self.padding.length, ...
                    self.adaptiveLowerBound, ...
                    self.adaptiveUpperBound );
            else
                self.fda.tSpanInput = self.fda.tSpanRegular;    
            end

            % process the input with a regularly-spaced time span
            [ self.XInputRegular, ~, self.XInputLen ] = ...
                               processXInput( ...
                                            XFd, ...
                                            XLen, ...
                                            self.fda.tSpan, ...
                                            self.fda.tSpanRegular, ...
                                            self.padding, ...
                                            true, ...
                                            length(self.fda.tSpanRegular), ...
                                            self.normalization );

            self.fda.fdParamsRegular = setFDAParameters( ...
                                            self.fda.tSpanRegular, ...
                                            self.fda.basisOrder, ...
                                            self.fda.penaltyOrder, ...
                                            self.fda.lambda );

            % process the input with an adaptive time span
            [ self.XInput, self.XInputDim ] = processXInput( ...
                                            XFd, ...
                                            XLen, ...
                                            self.fda.tSpan, ...
                                            self.fda.tSpanInput, ...
                                            self.padding, ...
                                            self.normalizeInput, ...
                                            length(self.fda.tSpanInput), ...
                                            self.normalization );

            self.fda.fdParamsInput = setFDAParameters( ...
                                            self.fda.tSpanInput, ...
                                            self.fda.basisOrder, ...
                                            self.fda.penaltyOrder, ...
                                            self.fda.lambda );


        end


        function self = prepareXTarget( self, XTargetRaw )
            % Prepare the target data based on input data
            % or a specified cell array
            arguments
                self                modelDataset
                XTargetRaw          cell
            end

            if isempty( XTargetRaw )
                % no new target dataset, use the same data as input

                self.fda.tSpanTarget = self.fda.tSpanInput;
                if self.normalizeInput
                    if self.matchingOutput
                        % matching input and output
                        self.XTarget = self.XInputRegular;
                    else
                        % time-normalize the input
                        self.XTarget = timeNormalize( self.XInput, ...
                                                      self.normalizedPts );
                    end

                else
                    % prepare normalized data of fixed length
                    pad = self.padding;
                    pad.length = max( self.XInputLen );
                    self.XTarget = normalizeXSeries( self.XInput, ...
                                                     self.normalizedPts, ...
                                                     self.normalization, ...
                                                     pad );
                end

                if self.matchingOutput
                    self.fda.tSpanTarget = self.fda.tSpanRegular;
                else
                    % adjust the time span in proportion to number of points
                    tSpan0 = 1:length( self.fda.tSpanInput );
                    tSpan1 = linspace( 1, length(self.fda.tSpanInput), ...
                                       self.normalizedPts );

                    self.fda.tSpanTarget= interp1( ...
                                tSpan0, self.fda.tSpanInput, tSpan1 );
                end


            else
                % new dataset requires processing
                targetX = processXSeries( self, XTargetRaw );
                % and time-normalized

                % !!!! This needs work !!!!

            end

            self.XTargetDim = size( self.XTarget, 1 );

            lambda = self.fda.lambda*self.fda.overSmoothing;
            self.fda.fdParamsTarget = setFDAParameters( ...
                                    self.fda.tSpanTarget, ...
                                    self.fda.basisOrder, ...
                                    self.fda.penaltyOrder, ...
                                    lambda );

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

            [ ds, XNfmt ] = createDatastore( self.XInput, self.XTarget, self.Y );

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


function [XFd, XLen] = smoothRawData( X, padding, fda )

    % find the series lengths (capped at padLen)
    XLen = min( cellfun( @length, X ), padding.length );

    % pad the series for smoothing
    X = padData( X, padding.length, padding.value, ...
                 Same = padding.same, ...
                 Location = padding.location, ...
                 Anchoring = padding.anchoring );
    
    % setup the smoothing parameters
    fdParams = setFDAParameters( fda.tSpan, ...
                                 fda.basisOrder, fda.penaltyOrder, ...
                                 fda.lambda );

    % create the smooth functions
    XFd = smooth_basis( fda.tSpan, X, fdParams );

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


function [ X, XDim, XLenNew ] = processXInput( ...
                                XFd, XLen, tSpan, tSpanNew, pad, ...
                                normalize, normalizedPts, normalization )

    % evaluate the input function at these points
    XEval = eval_fd( tSpanNew, XFd );
   
    % adjust lengths for non-linear re-sampling
    XLenNew = adjustXLengths( XLen, tSpan, tSpanNew, pad.location );
    
    % re-scale for resampled length
    XLenNew = ceil( size( XEval, 1 )*XLenNew / pad.length );
    pad.length = max( XLenNew );
    
    % recreate the cell time series
    XCell = extractXSeries( XEval, XLenNew, pad.length, pad.location );

    if normalize
        % use time-normalization method to set a fixed length
        XNorm = normalizeXSeries( XCell, normalizedPts, ...
                                        normalization, ...
                                        pad );
        XDim = size( XNorm, 1);
        X = num2cell( permute( XNorm, [2 1 3]), [2 3] );
        X = cellfun( @squeeze, X , 'UniformOutput', false);

    else
        % has variable length input
        X = XCell;
        XDim = 1;
    end

end


function tSpanAdaptive = calcAdaptiveTimeSpan( XFd, tSpan, ...
                                                resampleRate, padLen, ...
                                                lowerBound, upperBound )

    % resample the timespan
    tSpan = linspace( tSpan(1), tSpan(end), fix( padLen/resampleRate )+1 );   

    % evaluate the mean XFd curvature (2nd derivative)
    D1XEval = mean( abs(eval_fd( tSpan, XFd, 1 )), 2)';
    D2XEval = mean( abs(eval_fd( tSpan, XFd, 2 )), 2)';

    D1XEval = min( max( D1XEval/sum(D1XEval), lowerBound ), upperBound );
    D2XEval = min( max( D2XEval/sum(D2XEval), lowerBound ), upperBound );
    
    DXEvalComb = D1XEval + D2XEval;

    % cumulatively sum the absolute inverse curvatures
    % inserting zero at the begining to ensure first point will be at 0
    D2XInt = cumsum( [0 1./DXEvalComb] );
    D2XInt = D2XInt./max(D2XInt);

    % calculate variance as a function of time
    XVar = max( var( eval_fd( tSpan, XFd ), [], 2 ), 1E-2 );

    % cumulatively sum the absolute inverse curvatures
    XVarInt = cumsum( (1./XVar) );
    XVarInt = XVarInt./max(XVarInt);


    % normalize to the tSpan
    tSpanAdaptive = tSpan(1) + D2XInt*(tSpan(end)-tSpan(1));

    % reinterpolate to remove the extra point
    nPts = length(tSpan);
    tSpanAdaptive = interp1( 1:nPts+1, ...
                              tSpanAdaptive, ...
                              linspace(1, nPts+1, nPts) );

end


function XLen = adjustXLengths( XLen, tSpan, tSpanAdaptive, padding )

    for i = 1:length(XLen)
        switch padding

            case 'Left'
                tEnd = tSpan( length(tSpan)-XLen(i)+1 );
                XLen(i) = length(tSpan) - find( tEnd < tSpanAdaptive, 1 );

            case {'Right', 'Both'}
                tEnd = tSpan( XLen(i) );
                XLen(i) = find( tEnd < tSpanAdaptive, 1 );

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


function [ dsFull, XNfmt ] = createDatastore( X, XN, Y )

    % create the datastore for the input X
         
    % sort them in ascending order of length
    %XLen = cellfun( @length, X );
    %[ ~, orderIdx ] = sort( XLen, 'descend' );

    %X = X( orderIdx );
    dsX = arrayDatastore( X, 'IterationDimension', 1, ...
                             'OutputType', 'same' );
       
    % create the datastore for the time-normalised output X
    dsXN = arrayDatastore( XN, 'IterationDimension', 2 );
    if size( XN, 3 ) > 1
        XNfmt = 'SSCB';
    else
        XNfmt = 'CB';
    end
    
    % create the datastore for the labels/outcomes
    dsY = arrayDatastore( Y, 'IterationDimension', 1 );   
    
    % combine them
    dsFull = combine( dsX, dsXN, dsY );
               
end


function [ X, XN, Y ] = preprocMiniBatch( XCell, XNCell, YCell, ...
                                          padValue, padLoc )
    % Preprocess a sequence batch for training

    X = padData( XCell, 0, padValue, Longest = true, Location = padLoc  );
    %X = permute( X, [ 3 1 2 ] );
    
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
