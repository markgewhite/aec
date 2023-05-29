classdef ModelDataset
    % Class defining a dataset

    properties
        XLen            % array recording the length of each series

        XInputDim       % number of X dimensions for input
        XTargetDim      % number of X dimensions for output
        XChannels       % number of X input channels

        XFd             % functional data representation of raw data

        Y               % outcome variable

        Normalization       % structure for time-normalization 
        NormalizedPts       % standardized number of points for normalization
        HasNormalizedInput  % whether input should be time-normalized too

        Padding         % structure specifying padding setup
        FDA             % functional data analysis settings
        TSpan           % time span vector

        Perplexity          % for density estimation in X

        Info            % dataset information (used for plotting)
    end

    properties (Dependent = true)
        XInputCell      % processed input data as cell array
        XInput          % input data
        XTarget         % target output
        XTargetMean     % target output mean
        
        CDim            % number of categories
        YLabels         % Y labels
        NumObs          % number of observations
        NumObsByClass   % number of observation by class
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
                args.Normalization          char ...
                    {mustBeMember( args.Normalization, ...
                    {'PAD', 'LTN'} )} = 'LTN'
                args.NormalizedPts          double ...
                    {mustBeNumeric, mustBePositive, mustBeInteger} = 101
                args.PtsPerKnot             double ...
                    {mustBeGreaterThan( args.PtsPerKnot, 1 )} = 10
                args.HasNormalizedInput     logical = false
                args.Padding                struct ...
                    {mustBeValidPadding}
                args.BasisOrder             double ...
                    {mustBePositive, mustBeInteger} = 4
                args.PenaltyOrder           double ...
                    {mustBePositive, mustBeInteger} = 2
                args.Lambda                 double = []
                args.ResampleRate           double ...
                    {mustBeNumeric} = 1
                args.Perplexity             double ...
                    {mustBeNumeric} = 15
                args.DatasetName            string
                args.TimeLabel              string = "Time"
                args.ChannelLabels          string
                args.ClassLabels            string
                args.ChannelLimits          double
            end

            self.Y = Y;

            % set properties
            self.Normalization = args.Normalization;
            self.NormalizedPts = args.NormalizedPts;
            self.HasNormalizedInput = args.HasNormalizedInput;
            self.Padding = args.Padding;

            self.TSpan.Original = tSpan;
            self.TSpan.Input = linspace( ...
                                    self.TSpan.Original(1), ...
                                    self.TSpan.Original(end), ...
                                    fix( length(self.TSpan.Original/args.ResampleRate) ) );
            self.XInputDim = length( self.TSpan.Input );

            self.FDA.BasisOrder = args.BasisOrder;
            self.FDA.PenaltyOrder = args.PenaltyOrder;
            self.FDA.Lambda = args.Lambda;
            self.FDA.PtsPerKnot = args.PtsPerKnot;

            self.Perplexity = args.Perplexity;

            self.Info.DatasetName = args.DatasetName;
            self.Info.ChannelLabels = args.ChannelLabels;
            self.Info.TimeLabel = args.TimeLabel;
            self.Info.ClassLabels = args.ClassLabels;
            self.Info.ChannelLimits = args.ChannelLimits;

            self.XChannels = size( XInputRaw{1}, 2 );

            % create smooth functions for the data
            self = self.smoothRawData( XInputRaw );

        end


        function XCell = get.XInputCell( self )
            % Generate the regularly-spaced input from XFd 
            % returned as a cell array
            arguments
                self    ModelDataset
            end

            XCell = self.processX( self.TSpan.Input, ...
                                   true, ...
                                   length(self.TSpan.Input) );

        end


        function X = get.XInput( self )
            % Generate the regularly-spaced input from XFd
            arguments
                self    ModelDataset
            end

            XCell = self.processX( self.TSpan.Input, ...
                                   true, ...
                                   length(self.TSpan.Input) );

            X = reshape( cell2mat( XCell ), [], self.NumObs, self.XChannels );

        end


        function X = get.XTarget( self )
            % Generate the output from XFd
            % producing an array of fixed length
            arguments
                self    ModelDataset
            end

            XCell = self.processX( self.TSpan.Target, ...
                                   true, ...
                                   self.XTargetDim );

            X = reshape( cell2mat( XCell ), [], self.NumObs, self.XChannels );
            %X = timeNormalize( XCell, self.XTargetDim );

        end


        function XMean = get.XTargetMean( self )
            % Calculate the mean target curve
            arguments
                self            ModelDataset            
            end

            if self.XChannels == 1
                XMean = mean( self.XTarget, 2 );
            else
                XMean = mean( self.XTarget, 2 );
                XMean = permute( XMean, [1 3 2] );
            end

        end


        function YLabels = get.YLabels( self )
            % Generate the categories from Y
            arguments
                self            ModelDataset            
            end
               
            YLabels = categorical( unique(self.Y) );

        end


        function CDim = get.CDim( self )
            % Get the number of classes
            arguments
                self            ModelDataset            
            end
               
            CDim = length( self.YLabels );

        end


        function NumObs = get.NumObs( self )
            % Get the number of classes
            arguments
                self            ModelDataset            
            end
               
            NumObs = length( self.Y );

        end


        function NumObsByClass = get.NumObsByClass( self )
            % Get the number of observations by class
            arguments
                self            ModelDataset            
            end
               
            NumObsByClass = groupcounts( self.Y );

        end
        
        
        % class methods

        dsFull = getDatastore( self )

        selection = getCVPartition( self, args )

        mbq = getMiniBatchQueue( self, batchSize, XLabels, XNLabels, args )

        unit = getPartitioningUnit( self )

        thisSubset = partition( self, idx )

        fig = plot( self, args )

        [ fdParams, lambda ] = setTargetFdParams( self, X )

    end

end