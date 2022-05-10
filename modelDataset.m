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

        Y               % outcome variable
        YDim            % number of categories
        YLabels         % Y labels
        nObs            % number of observations

        normalization   % structure for time-normalization 
        normalizedPts   % standardized number of points for normalization
        normalizeInput  % whether input should be time-normalized too

        padding         % structure specifying padding setup

        fda             % functional data analysis settings
        adaptiveTimeSpan % flag whether to adjust timespan to complexity
        resampleRate    % downsampling rate

        info            % dataset information (used for plotting)
    end


    methods

        function self = modelDataset( XInputRaw, Y, args )
            % Create and preprocess the data.
            % The calling function will be a data loader or
            % a function partitioning the data.
            arguments
                XInputRaw               cell
                Y
                args.XTargetRaw         cell = []
                args.purpose            char ...
                    {mustBeMember( args.purpose, ...
                    {'Creation', 'ForSubset'} )} = 'Creation'
                args.normalization      char ...
                    {mustBeMember( args.normalization, ...
                    {'PAD', 'LTN'} )} = 'LTN'
                args.normalizedPts      double ...
                    {mustBeNumeric, mustBePositive, mustBeInteger} = 101
                args.normalizeInput     logical = false;
                args.padding            struct ...
                    {mustBeValidPadding}
                args.fda                struct ...
                    {mustBeValidFdParams}
                args.adaptiveTimeSpan   logical = false
                args.resampleRate       double ...
                    {mustBeNumeric} = 1
                args.datasetName        string
                args.timeLabel          string = "Time"
                args.channelLabels      string
                args.channelLimits      double
            end

            self.XInputRaw = XInputRaw;
            self.Y = Y;
            
            if strcmp( args.purpose, 'ForSubset' )
                % don't do the setup
                return
            end

            self.normalization = args.normalization;
            self.normalizedPts = args.normalizedPts;
            self.normalizeInput = args.normalizeInput;
            self.padding = args.padding;
            self.fda = args.fda;
            self.adaptiveTimeSpan = args.adaptiveTimeSpan;
            self.resampleRate = args.resampleRate;

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
            self.YDim = length( self.YLabels );          

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

            thisSubset = modelDataset( subXRaw, subY, ...
                                       purpose='ForSubset' );

            thisSubset.normalization = self.normalization;
            thisSubset.normalizedPts = self.normalizedPts;
            thisSubset.normalizeInput = self.normalizeInput;

            thisSubset.padding = self.padding;
            thisSubset.fda = self.fda;
            thisSubset.resampleRate = self.resampleRate;

            thisSubset.XInput = split( self.XInput, idx );
            thisSubset.XTarget = split( self.XTarget, idx );
            thisSubset.XInputLen = self.XInputLen( idx );

            thisSubset.XInputDim = self.XInputDim;
            thisSubset.XTargetDim = self.XTargetDim;
            thisSubset.XChannels = self.XChannels;

            thisSubset.YDim = self.YDim;
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
            
            if iscell( self.XInput )
                X = padData( self.XInput, 'Longest', ...
                               self.padding.value, self.padding.location );
                X = permute( X, [ 3 1 2 ] );
                if arg.dlarray
                    X = dlarray( X, 'CTB' );
                end

            else
                if arg.dlarray
                    X = dlarray( self.XInput, 'CB' );
                else
                    X = self.XInput;
                end

            end

            if arg.dlarray
                Y = dlarray( self.Y, 'CB' );
            else
                Y = self.Y;
            end

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
                    fix( self.padding.length/self.resampleRate )+1 );  

            % adaptive resampling, as required
            if self.adaptiveTimeSpan
                self.fda.tSpanInput = calcAdaptiveTimeSpan( ...
                    XFd, self.fda.tSpan, ...
                    self.resampleRate, self.padding.length );
            else
                self.fda.tSpanInput = self.fda.tSpanRegular;    
            end

            % process the input with a regularly-spaced time span
            [ self.XInputRegular, self.XInputDim, self.XInputLen ] = ...
                               processXInput( ...
                                            XFd, ...
                                            XLen, ...
                                            self.fda.tSpan, ...
                                            self.fda.tSpanRegular, ...
                                            self.padding, ...
                                            self.normalizeInput, ...
                                            self.normalizedPts, ...
                                            self.normalization );

            self.fda.fdParamsRegular = setFDAParameters( ...
                                            self.fda.tSpanRegular, ...
                                            self.fda.basisOrder, ...
                                            self.fda.penaltyOrder, ...
                                            self.fda.lambda );

            % process the input with an adaptive time span
            self.XInput  = processXInput( ...
                                            XFd, ...
                                            XLen, ...
                                            self.fda.tSpan, ...
                                            self.fda.tSpanInput, ...
                                            self.padding, ...
                                            self.normalizeInput, ...
                                            self.normalizedPts, ...
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
                    % re-use the time-normalized input
                    self.XTarget = self.XInput;

                else
                    % prepare normalized data of fixed length
                    self.XTarget = normalizeXSeries( self.XInput, ...
                                                     self.normalizedPts, ...
                                                     self.normalization, ...
                                                     self.padding );
                end

                % adjust the time span in proportion to number of points
                tSpan0 = 1:length( self.fda.tSpanInput );

                tSpan1 = linspace( 1, length(self.fda.tSpanInput), ...
                                   self.normalizedPts );

                self.fda.tSpanTarget= interp1( ...
                                tSpan0, self.fda.tSpanInput, tSpan1 );


            else
                % new dataset requires processing
                targetX = processXSeries( self, XTargetRaw );
                % and time-normalized

                % !!!! This needs work !!!!

            end

            self.XTargetDim = size( self.XTarget, 1 );

            self.fda.fdParamsTarget = setFDAParameters( ...
                                            self.fda.tSpanTarget, ...
                                            self.fda.basisOrder, ...
                                            self.fda.penaltyOrder, ...
                                            self.fda.lambda );

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

            % setup the minibatch queues
            if self.isFixedLength
                % no preprocessing required

                mbq = minibatchqueue( ds,...
                                  'MiniBatchSize', batchSize, ...
                                  'PartialMiniBatch', args.partialBatch, ...
                                  'MiniBatchFormat', {'CB', XNfmt, 'BC'} );
            
            else
                % setup the minibatch preprocessing function
                preproc = @( X, XN, Y ) preprocMiniBatch( X, XN, Y, ...
                              self.padding.value, ...
                              self.padding.location );

                mbq = minibatchqueue(  ds,...
                                  'MiniBatchSize', batchSize, ...
                                  'PartialMiniBatch', args.partialBatch, ...
                                  'MiniBatchFcn', preproc, ...
                                  'MiniBatchFormat', {'CTB', XNfmt, 'CB'} );

            end

        end


    end

end


function [XFd, XLen] = smoothRawData( X, padding, fda )

    % find the series lengths (capped at padLen)
    XLen = min( cellfun( @length, X ), padding.length );

    % pad the series for smoothing
    X = padData( X, padding.length, padding.value, padding.location );
    
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


function [ XNew, XDim, XLenNew ] = processXInput( ...
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
        XNew = normalizeXSeries( XCell, normalizedPts, ...
                                        normalization, ...
                                        pad );
        XDim = size( XNew, 1);

    else
        % has variable length input
        XNew = XCell;
        XDim = 1;
    end

end


function tSpanAdaptive = calcAdaptiveTimeSpan( XFd, tSpan, ...
                                                resampleRate, padLen )

    % resample the timespan
    tSpan = linspace( tSpan(1), tSpan(end), fix( padLen/resampleRate )+1 );   

    % evaluate the mean XFd curvature (2nd derivative)
    D1XEval = mean(max( abs(eval_fd( tSpan, XFd, 1 )), 1E-8 ), 2)';
    D2XEval = mean(max( abs(eval_fd( tSpan, XFd, 2 )), 1E-8 ), 2)';
    DXEvalComb = D1XEval/sum(D1XEval) + D2XEval/sum(D2XEval);

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

            case 'left'
                tEnd = tSpan( length(tSpan)-XLen(i)+1 );
                XLen(i) = length(tSpan) - find( tEnd < tSpanAdaptive, 1 );

            case {'right', 'both'}
                tEnd = tSpan( XLen(i) );
                XLen(i) = find( tEnd < tSpanAdaptive, 1 );

        end
    end

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


function XN = normalizeXSeries( X, nPts, type, pad )

    switch type
    
        case 'LTN' % time normalization
            XN = timeNormalize( X, nPts );

        case 'PAD' % padding
            XN = padData( X, pad.length, pad.value, pad.location );
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
    if iscell( X )           
        % sort them in ascending order of length
        XLen = cellfun( @length, X );
        [ ~, orderIdx ] = sort( XLen, 'descend' );
    
        X = X( orderIdx );
        dsX = arrayDatastore( X, 'IterationDimension', 1, ...
                                 'OutputType', 'same' );
    
    else
        dsX = arrayDatastore( X, 'IterationDimension', 2 );
    
    end
    
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

    X = padData( XCell, 'Longest', padValue, padLoc );
    X = permute( X, [ 3 1 2 ] );
    
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
