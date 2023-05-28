classdef ModelDataset
    % Class defining a dataset

    properties
        XLen            % array recording the length of each series

        XInputDim       % number of X input dimensions
        XTargetDim      % normalized X output size (time series length)
        XChannels       % number of X input channels

        XFd             % functional data representation of raw data

        Y               % outcome variable

        Normalization       % structure for time-normalization 
        NormalizedPts       % standardized number of points for normalization
        HasNormalizedInput  % whether input should be time-normalized too
        HasMatchingOutput   % whether input and target output must match

        Padding         % structure specifying padding setup
        FDA             % functional data analysis settings

        TSpan           % time span structure of vectors holding
                        %   .Original = matching the raw data
                        %   .Regular = regularly-spaced times
                        %   .Input = input to the model
                        %   .Target = for autoencoder reconstructions

        ResampleRate        % downsampling rate
        Perplexity          % for density estimation in X

        Info            % dataset information (used for plotting)
    end

    properties (Dependent = true)
        XInputCell      % processed input data (variable length) as cell array
        XInputRegular   % processed input with regularly spaced time span
        XInputCoeff     % input Fd coefficients as cell array
        XInputCoeffRegular  % input Fd coefficients as array
        XInputCoeffDim  % Fd input dimensions
        XTarget         % target output
        XTargetMean     % target output mean
        XTargetCoeff    % target Fd coefficients
        XTargetCoeffDim % Fd target dimensions
        XTargetCoeffMean % target output mean
        
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
                args.normalization          char ...
                    {mustBeMember( args.normalization, ...
                    {'PAD', 'LTN'} )} = 'LTN'
                args.normalizedPts          double ...
                    {mustBeNumeric, mustBePositive, mustBeInteger} = 101
                args.ptsPerKnot             double ...
                    {mustBeGreaterThan( args.ptsPerKnot, 1 )} = 10
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

            self.Y = Y;

            % set properties
            self.Normalization = args.normalization;
            self.NormalizedPts = args.normalizedPts;
            self.HasNormalizedInput = args.hasNormalizedInput;
            self.HasMatchingOutput = args.hasMatchingOutput;
            self.Padding = args.padding;

            self.FDA.BasisOrder = args.basisOrder;
            self.FDA.PenaltyOrder = args.penaltyOrder;
            self.FDA.Lambda = args.lambda;
            self.FDA.PtsPerKnot = args.ptsPerKnot;

            self.TSpan.Original = tSpan;
            self.ResampleRate = args.resampleRate;
            self.Perplexity = args.perplexity;

            self.Info.DatasetName = args.datasetName;
            self.Info.ChannelLabels = args.channelLabels;
            self.Info.TimeLabel = args.timeLabel;
            self.Info.ClassLabels = args.classLabels;
            self.Info.ChannelLimits = args.channelLimits;

            self.XChannels = size( XInputRaw{1}, 2 );

            % create smooth functions for the data
            [self.XFd, self.XLen, self.FDA.Lambda] = self.smoothRawData( XInputRaw );

            % re-sampling at the given rate
            self.TSpan.Regular = linspace( ...
                    self.TSpan.Original(1), ...
                    self.TSpan.Original(end), ...
                    fix( self.Padding.Length/self.ResampleRate ) );  

            if isempty( args.tSpan )
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

            % set the FD parameters for regular spacing
            self.FDA.FdParamsRegular = self.setFDAParameters( self.TSpan.Regular);

            % set the FD parameters for input
            self.FDA.FdParamsInput = self.setFDAParameters( self.TSpan.Input );

            % set the FD parameters for the target
            self.FDA.FdParamsTarget = self.setFDAParameters( self.TSpan.Target );
            
            % set the FD parameters for the components to the same values for now 
            self.FDA.FdParamsComponent = self.FDA.FdParamsTarget;

            % set input and output dimensions
            self.XInputDim = length( self.TSpan.Input );
            if self.HasMatchingOutput
                self.XTargetDim = self.XInputDim;
            else 
                self.XTargetDim = self.NormalizedPts;
            end

        end


        function X = get.XInputCoeff( self )
            % Get the Fd coefficients for the input
            arguments
                self    ModelDataset
            end

            X = eval_fd( self.TSpan.Input, self.XFd ); 
            XInputFd = smooth_basis( self.TSpan.Input, ...
                                X, ...
                                self.FDA.FdParamsInput );
            XCoeff = single(getcoef( XInputFd ));

            if size( XCoeff, 3 ) > 1
                X = num2cell( permute( XCoeff, [2 1 3]), [2 3] );
                X = cellfun( @squeeze, X , 'UniformOutput', false);
            else
                X = num2cell( permute( XCoeff, [2 1]), 2 );
                X = cellfun( @transpose, X , 'UniformOutput', false);
            end

        end


        function XCoeff = get.XInputCoeffRegular( self )
            % Get the Fd coefficients for the input
            arguments
                self    ModelDataset
            end

            X = eval_fd( self.TSpan.Input, self.XFd ); 
            XInputFd = smooth_basis( self.TSpan.Input, ...
                                X, ...
                                self.FDA.FdParamsRegular );
            XCoeff = single(getcoef( XInputFd ));

        end


        function d = get.XInputCoeffDim( self )
            % Get dimension of target coefficients
            arguments
                self    ModelDataset
            end

            d = length(getcoef( self.FDA.FdParamsInput ));

        end


        function X = get.XInputCell( self )
            % Generate cell array input from XFd
            arguments
                self    ModelDataset
            end

            X = processX(  self.XFd, ...
                           self.XLen, ...
                           self.TSpan.Original, ...
                           self.TSpan.Regular, ...
                           self.Padding, ...
                           self.HasNormalizedInput, ...
                           length(self.TSpan.Input), ...
                           self.Normalization );

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
        
        
        function XCoeff = get.XTargetCoeff( self )
            % Get the Fd coefficients for the target
            arguments
                self    ModelDataset
            end

            X = eval_fd( self.TSpan.Target, self.XFd ); 
            XTargetFd = smooth_basis( self.TSpan.Target, ...
                                X, ...
                                self.FDA.FdParamsTarget );
            XCoeff = single(getcoef( XTargetFd ));

        end


        function d = get.XTargetCoeffDim( self )
            % Get dimension of target coefficients
            arguments
                self    ModelDataset
            end

            d = length(getcoef( self.FDA.FdParamsTarget ));

        end


        function X = get.XTarget( self )
            % Generate the output from XFd
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
     

        function XMean = get.XTargetCoeffMean( self )
            % Calculate the mean target Fd coefficients
            arguments
                self            ModelDataset            
            end

            if self.XChannels == 1
                XMean = mean( self.XTargetCoeff, 2 );
            else
                XMean = mean( self.XTargetCoeff, 2 );
                XMean = permute( XMean, [1 3 2] );
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