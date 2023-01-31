classdef ModelDataset
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
        NumObsByClass   % number of observation by class

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
        Perplexity          % for density estimation in X

        Info            % dataset information (used for plotting)
    end

    properties (Dependent = true)
        XInput          % processed input data (variable length)
        XTarget         % target output
        XTargetMean     % target output mean
        XInputRegular   % processed input with regularly spaced time span
        XInputRegularMean % mean of the processed regular input curve
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
                args.basisOrder             double ...
                    {mustBePositive, mustBeInteger} = 4
                args.penaltyOrder           double ...
                    {mustBePositive, mustBeInteger} = 2
                args.lambda                 double = []
                args.tSpan                  double = []
                args.hasAdaptiveTimeSpan    logical = false
                args.adaptiveLowerBound     double = 0.05
                args.adaptiveUpperBound     double = 5
                args.resampleRate           double ...
                    {mustBeNumeric} = 1
                args.perplexity             double ...
                    {mustBeNumeric} = 15
                args.datasetName            string
                args.timeLabel              string = "Time"
                args.channelLabels          string
                args.classLabels            string
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

            self.FDA.BasisOrder = args.basisOrder;
            self.FDA.PenaltyOrder = args.penaltyOrder;
            self.FDA.Lambda = args.lambda;

            self.TSpan.Original = tSpan;
            self.HasAdaptiveTimeSpan = args.hasAdaptiveTimeSpan;
            self.AdaptiveLowerBound = args.adaptiveLowerBound;
            self.AdaptiveUpperBound = args.adaptiveUpperBound;
            self.ResampleRate = args.resampleRate;
            self.Perplexity = args.perplexity;

            self.Info.DatasetName = args.datasetName;
            self.Info.ChannelLabels = args.channelLabels;
            self.Info.TimeLabel = args.timeLabel;
            self.Info.ClassLabels = args.classLabels;
            self.Info.ChannelLimits = args.channelLimits;

            % get immediately available dimensions
            self.NumObs = length( XInputRaw );
            self.XChannels = size( XInputRaw{1}, 2 );

            % create smooth functions for the data
            [self.XFd, self.XLen, self.FDA.Lambda] = self.smoothRawData( XInputRaw );

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
            self.FDA.FdParamsRegular = self.setFDAParameters( self.TSpan.Regular);

            % set the FD parameters for adaptive spacing
            self.FDA.FdParamsInput = self.setFDAParameters( self.TSpan.Input );

            % set the FD parameters for adaptive spacing
            self.FDA.FdParamsTarget = self.setFDAParameters( self.TSpan.Target );

            % assign category labels
            self.YLabels = categorical( unique(self.Y) );
            self.CDim = length( self.YLabels );
            self.NumObsByClass = groupcounts( self.Y );

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


        function XMean = get.XInputRegularMean( self )
            % Calculate the mean input regular curve
            arguments
                self            ModelDataset            
            end

            if self.XChannels == 1
                XMean = mean( self.XInputRegular, 2 );
            else
                XMean = mean( self.XInputRegular, 3 );
            end

        end
     

        function XMean = get.XTargetMean( self )
            % Calculate the mean target curve
            arguments
                self            ModelDataset            
            end

            if self.XChannels == 1
                XMean = mean( self.XTarget, 2 );
            else
                XMean = mean( self.XTarget, 3 );
            end

        end


        % class methods

        dsFull = getDatastore( self )

        selection = getCVPartition( self, args )

        [ X, Y ] = getDLInput( self, labels, arg )

        mbq = getMiniBatchQueue( self, batchSize, XLabels, XNLabels, args )

        unit = getPartitioningUnit( self )

        thisSubset = partition( self, idx )

        fig = plot( self, args )

        [ fdParams, lambda ] = setTargetFdParams( self, X )

    end

end