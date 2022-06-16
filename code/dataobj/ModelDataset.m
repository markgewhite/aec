classdef ModelDataset < handle
    % Class defining a dataset

    properties
        XInputRaw       % original, raw time series data
        XFd             % functional representation of the data
        XLen            % array recording the length of each series

        XInputDim       % number of X input dimensions
        XTargetDim      % normalized X output size (time series length)
        XChannels       % number of X input channels

        Y               % outcome variable
        CDim            % number of categories
        YLabels         % Y labels
        NumObs          % number of observations

        Normalization       % structure for time-normalization 
        NormalizedPts       % standardized number of points for normalization
        HasNormalizedInput  % whether input should be time-normalized too
        HasMatchingOutput   % whether input and target output must match

        Padding         % structure specifying padding setup
        FDA             % functional data analysis settings

        TSpan           % time span structure of vectors holding
                        %   .Original = matching the raw data
                        %   .Regular = regularly-spaced times
                        %   .Input = input to the model (may be adaptive)
                        %   .Target = for autoencoder reconstructions

        HasAdaptiveTimeSpan % flag whether to adjust timespan to complexity
        AdaptiveLowerBound  % lower limit on point density
        AdaptiveUpperBound  % upper limit on point density
        ResampleRate        % downsampling rate
        OverSmoothing       % additional factor for target roughness penalty 

        Info            % dataset information (used for plotting)
    end

    properties (Dependent = true)
        XInput          % processed input data (variable length)       
        XTarget         % target output
        XInputRegular   % processed input with regularly spaced time span
    end


    methods

        function self = ModelDataset( XInputRaw, Y, tSpan, args )
            % Create and preprocess the data.
            % The calling function will be a data loader or
            % a function partitioning the data.
            arguments
                XInputRaw                   cell
                Y
                tSpan                       double
                args.XTargetRaw             cell = []
                args.purpose                char ...
                    {mustBeMember( args.purpose, ...
                    {'Creation', 'ForSubset'} )} = 'Creation'
                args.normalization          char ...
                    {mustBeMember( args.normalization, ...
                    {'PAD', 'LTN'} )} = 'LTN'
                args.normalizedPts          double ...
                    {mustBeNumeric, mustBePositive, mustBeInteger} = 101
                args.hasNormalizedInput     logical = false
                args.hasMatchingOutput      logical = false
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
            
            if strcmp( args.purpose, 'ForSubset' )
                % don't do the setup
                return
            end

            % set properties
            self.Normalization = args.normalization;
            self.NormalizedPts = args.normalizedPts;
            self.HasNormalizedInput = args.hasNormalizedInput;
            self.HasMatchingOutput = args.hasMatchingOutput;
            self.Padding = args.padding;
            self.FDA = args.fda;

            self.TSpan.Original = tSpan;
            self.HasAdaptiveTimeSpan = args.hasAdaptiveTimeSpan;
            self.AdaptiveLowerBound = args.adaptiveLowerBound;
            self.AdaptiveUpperBound = args.adaptiveUpperBound;
            self.ResampleRate = args.resampleRate;
            self.OverSmoothing = args.overSmoothing;

            self.Info.DatasetName = args.datasetName;
            self.Info.ChannelLabels = args.channelLabels;
            self.Info.TimeLabel = args.timeLabel;
            self.Info.ChannelLimits = args.channelLimits;

            % get immediately available dimensions
            self.NumObs = length( XInputRaw );
            self.XChannels = size( XInputRaw{1}, 2 );

            % create smooth functions for the data
            [self.XFd, self.XLen] = smoothRawData( XInputRaw, ...
                                                   self.Padding, ...
                                                   self.TSpan.Original, ...
                                                   self.FDA  );

            % re-sampling at the given rate
            self.TSpan.Regular = linspace( ...
                    self.TSpan.Original(1), ...
                    self.TSpan.Original(end), ...
                    fix( self.Padding.Length/self.ResampleRate ) );  

            % adaptive resampling, as required
            if self.HasAdaptiveTimeSpan && isempty( args.tSpan )

                self.TSpan.Input = calcAdaptiveTimeSpan( ...
                    self.XFd, ...
                    self.TSpan.Regular, ...
                    self.AdaptiveLowerBound, ...
                    self.AdaptiveUpperBound );

            elseif isempty( args.tSpan )
                self.TSpan.Input = self.TSpan.Regular;
            else
                self.TSpan.Input = args.tSpan;
            end

            if self.HasMatchingOutput
                % ensure the input and target time spans match
                % this usually applies for PCA
                self.TSpan.Target = self.TSpan.Regular;

            else
                % adjust the time span in proportion to number of points
                tSpan0 = 1:length( self.TSpan.Input );
                tSpan1 = linspace( 1, length(self.TSpan.Input), ...
                                   self.NormalizedPts );

                self.TSpan.Target= interp1( ...
                            tSpan0, self.TSpan.Input, tSpan1 );
            end

            self.XInputDim = length( self.TSpan.Input );
            if self.HasMatchingOutput
                self.XTargetDim = self.XInputDim;
            else
                self.XTargetDim = self.NormalizedPts;
            end

            % set the FD parameters for regular spacing
            self.FDA.FdParamsRegular = setFDAParameters( ...
                                            self.TSpan.Regular, ...
                                            self.FDA.BasisOrder, ...
                                            self.FDA.PenaltyOrder, ...
                                            self.FDA.Lambda );

            % set the FD parameters for adaptive spacing
            self.FDA.FdParamsInput = setFDAParameters( ...
                                            self.TSpan.Input, ...
                                            self.FDA.BasisOrder, ...
                                            self.FDA.PenaltyOrder, ...
                                            self.FDA.Lambda );

            % set the FD parameters for adaptive spacing
            % with a higher level of smoothing than the input
            lambda = self.FDA.Lambda*self.OverSmoothing;
            self.FDA.FdParamsTarget = setFDAParameters( ...
                                    self.TSpan.Target, ...
                                    self.FDA.BasisOrder, ...
                                    self.FDA.PenaltyOrder, ...
                                    lambda );

            % assign category labels
            self.YLabels = categorical( unique(self.Y) );
            self.CDim = length( self.YLabels );          

        end


        function thisSubset = partition( self, idx )
            % Create the subset of this ModelDataset
            % using the indices specified
            arguments
                self        ModelDataset
                idx         logical 
            end

            subXRaw = split( self.XInputRaw, idx );
            subY = self.Y( idx );
            subTSpan = self.TSpan.Original;

            thisSubset = ModelDataset( subXRaw, subY, subTSpan, ...
                                       purpose = 'ForSubset' );

            thisSubset.XFd = splitFd( self.XFd, idx );
            thisSubset.XLen = self.XLen( idx );

            thisSubset.Normalization = self.Normalization;
            thisSubset.NormalizedPts = self.NormalizedPts;
            thisSubset.HasNormalizedInput = self.HasNormalizedInput;

            thisSubset.Padding = self.Padding;
            thisSubset.FDA = self.FDA;
            thisSubset.ResampleRate = self.ResampleRate;

            thisSubset.XInputDim = self.XInputDim;
            thisSubset.XTargetDim = self.XTargetDim;
            thisSubset.XChannels = self.XChannels;

            thisSubset.CDim = self.CDim;
            thisSubset.YLabels = self.YLabels;
            thisSubset.NumObs = sum( idx );

            thisSubset.TSpan = self.TSpan;

            thisSubset.HasMatchingOutput = self.HasMatchingOutput;
            thisSubset.HasAdaptiveTimeSpan = self.HasAdaptiveTimeSpan;
            thisSubset.AdaptiveLowerBound = self.AdaptiveLowerBound;
            thisSubset.AdaptiveUpperBound = self.AdaptiveUpperBound;

            thisSubset.Info = self.Info;

        end


        function isFixed = isFixedLength( self )
            % return whether data is time-normalized
            arguments
                self    ModelDataset
            end

            isFixed = self.HasNormalizedInput;
            
        end



        function [ X, Y ] = getDLInput( self, labels, arg )
            % Convert X and Y into dl arrays
            arguments
                self            ModelDataset
                labels          char
                arg.dlarray     logical = true
            end
            
            X = padData( self.XInput, 0, self.Padding.Value, ...
                         Longest = true, ...
                         Same = self.Padding.Same, ...
                         Location = self.Padding.Location );

            X = dlarray( X, labels );

            if arg.dlarray
                Y = dlarray( self.Y, 'CB' );
            else
                Y = self.Y;
            end

        end


        function validateSmoothing( self, args )
            % Re-run smoothing with maximum flexibility
            arguments
                self            ModelDataset
                args.X          double = []
                args.TSpan      double = []
            end

            if isempty( args.X )
                % default: use the raw input series
                % pad the raw series for smoothing
                X = padData( self.XInputRaw, ...
                             self.Padding.Length, ...
                             self.Padding.Value, ...
                             Same = self.Padding.Same, ...
                             Location = self.Padding.Location, ...
                             Anchoring = self.Padding.Anchoring );
            else
                % use the specified X
                X = args.X;

            end

            if isempty( args.TSpan )
                % default: use the orginal time series
                % create a time span with maximum detail
                thisTSpan = linspace( self.TSpan.Original(1),...
                              self.TSpan.Original(end), ...
                              size( X, 1 ) );
            else
                % use the specified timespan
                thisTSpan = args.TSpan;
            end

            % create a new basis with maximum number of functions
            basis = create_bspline_basis( [thisTSpan(1) thisTSpan(end)], ...
                                          size( X, 1 ), ...
                                          self.FDA.BasisOrder);
            
            % Find minimum GCV value of lambda
            % search for the best value for lambda, the roughness penalty
            logLambda   = -10:1:10;
            gcvSave = zeros( length(logLambda), self.XChannels );
            dfSave  = zeros( length(logLambda), 1 );
            
            for i = 1:length(logLambda)
                
                % set smoothing parameters
                XFdPari = fdPar( basis, ...
                                 self.FDA.PenaltyOrder, ...
                                 10^logLambda(i) );
                
                % perform smoothing
                [~, dfi, gcvi] = smooth_basis( thisTSpan, X, XFdPari );
                
                % determine mean GCV and degrees of freedom
                gcvSave(i,:) = sqrt( sum( gcvi, 2 )/self.NumObs ); 
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
                self    ModelDataset
            end

            XCell = processX( self.XFd, ...
                               self.XLen, ...
                               self.TSpan.Original, ...
                               self.TSpan.Regular, ...
                               self.Padding, ...
                               true, ...
                               length(self.TSpan.Regular), ...
                               self.Normalization );

            X = reshape( cell2mat( XCell ), [], self.NumObs, self.XChannels );

        end


        function X = get.XInput( self )
            % Generate the adaptively-spaced input from XFd
            arguments
                self    ModelDataset
            end

            X = processX(  self.XFd, ...
                           self.XLen, ...
                           self.TSpan.Original, ...
                           self.TSpan.Input, ...
                           self.Padding, ...
                           self.HasNormalizedInput, ...
                           length(self.TSpan.Input), ...
                           self.Normalization );

        end
        

        function X = get.XTarget( self )
            % Generate the adaptively-spaced input from XFd
            % producing an array of fixed length
            arguments
                self    ModelDataset
            end

            if self.HasMatchingOutput
                numPts = self.XInputDim;
            else
                numPts = self.NormalizedPts;
            end

            XCell = processX(  self.XFd, ...
                               self.XLen, ...
                               self.TSpan.Original, ...
                               self.TSpan.Target, ...
                               self.Padding, ...
                               true, ...
                               numPts, ...
                               self.Normalization );

            X = timeNormalize( XCell, numPts );

        end


        function unit = getPartitioningUnit( self )
            % Provide the variable to be unit for partitioning
            % Placeholder function that may be overridden by children
            arguments
                self    ModelDataset
            end

            unit = 1:self.NumObs;

        end
        

        function selection = getCVPartition( self, args )
            % Generate a CV partition for the dataset
            arguments
                self                ModelDataset
                args.Holdout        double ...
                    {mustBeInRange(args.Holdout, 0, 1)}
                args.KFold          double ...
                    {mustBeInteger, mustBePositive}
                args.Identical      logical = false
            end

            if ~isfield( args, 'Holdout' ) && ~isfield( args, 'KFold' )
                eid = 'ModelDataset:PartitioningNotSpecified';
                msg = 'Partitioning scheme not specified.';
                throwAsCaller( MException(eid,msg) );
            end

            if isfield( args, 'Holdout' ) && isfield( args, 'KFold' )
                eid = 'ModelDataset:PartitioningOverSpecified';
                msg = 'Two partitioning schemes specified, not one.';
                throwAsCaller( MException(eid,msg) );
            end

            unit = self.getPartitioningUnit;
            uniqueUnit = unique( unit );

            if isfield( args, 'Holdout' )
                % holdout partitioning
                cvpart = cvpartition( length( uniqueUnit ), ...
                                      Holdout = args.Holdout );
                selection = ismember( unit, uniqueUnit( training(cvpart) ));
              
            else
                % K-fold partitioning
                cvpart = cvpartition( length( uniqueUnit ), ...
                                      KFold = args.KFold );
                
                if length( uniqueUnit ) < length( unit )
                    % partitioning unit is a grouping variable
                    selection = false( self.NumObs, args.KFold );
                    for k = 1:args.KFold
                        if args.Identical
                            % special case - make all partitions the same
                            f = 1;
                        else
                            f = k;
                        end
                        selection( :, k ) = ismember( unit, ...
                                        uniqueUnit( training(cvpart,f) ));
                    end
                else
                    selection = training( cvpart );
                end

            end

        end


        function mbq = getMiniBatchQueue( self, batchSize, XLabels, XNLabels, args )
            % Create a minibatch queue
            arguments
                self                ModelDataset
                batchSize           double ...
                    {mustBeInteger, mustBePositive}
                XLabels             char
                XNLabels            char
                args.partialBatch   char ...
                    {mustBeMember( args.partialBatch, ...
                    {'return', 'discard'} )} = 'discard'
            end

            ds = createDatastore( self.XInput, self.XTarget, self.Y );

            % setup the minibatch preprocessing function
            preproc = @( X, XN, Y ) preprocMiniBatch( X, XN, Y, ...
                          self.Padding.Value, ...
                          self.Padding.Location );

            mbq = minibatchqueue(  ds,...
                  MiniBatchSize = batchSize, ...
                  PartialMiniBatch = args.partialBatch, ...
                  MiniBatchFcn = preproc, ...
                  MiniBatchFormat = {XLabels, XNLabels, 'CB'} );

        end


    end

end


function [XFd, XLen] = smoothRawData( XCell, padding, tSpan, fda )

    % find the series lengths (capped at padLen)
    XLen = min( cellfun( @length, XCell ), padding.Length );

    % pad the series for smoothing
    X = padData( XCell, padding.Length, padding.Value, ...
                 Same = padding.Same, ...
                 Location = padding.Location, ...
                 Anchoring = padding.Anchoring );
    
    % setup the smoothing parameters
    fdParams = setFDAParameters( tSpan, ...
                                 fda.BasisOrder, fda.PenaltyOrder, ...
                                 fda.Lambda );

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
    XLenNew = adjustXLengths( XLen, tSpan, tSpanNew, pad.Location );
    
    % re-scale for resampled length
    pad.Length = max( XLenNew );
    
    % recreate the cell time series
    XCell = extractXSeries( XEval, XLenNew, pad.Length, pad.Location );

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
            XN = padData( X, pad.Length, pad.Value, ...
                             Same = pad.Same, ...
                             Location = pad.Location, ...
                             Anchoring = pad.Anchoring );
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
