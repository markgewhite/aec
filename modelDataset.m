classdef modelDataset
    % Class defining a model trainer

    properties
        XRaw            % original, raw time series data
        X               % processed input data (variable length)
        XN              % time-normalized data, the target output
        Y               % outcome variable
        XDim            % X input size (array of time series lengths)
        XNDim           % normalized X output size (time series length)
        XChannels       % number of X channels
        YDim            % number of categories
        YLabels         % Y labels
        nObs            % number of observations

        normalization   % type of normalization for output
        normalizedPts   % standardized number of points for normalization
        normalizeInput  % whether to also normalize the input
        padding         % structure specifying padding setup
        fda             % functional data analysis settings
        resampleRate    % downsampling rate
        hasDerivative   % if the data includes first derivative
    end
    properties (Dependent)
        Z               % latent codes from the model
        XC              % components data from the model
    end



    methods

        function self = modelDataset( XRaw, Y, args )
            % Create and preprocess the data.
            % The calling function will be a data loader or
            % a function partitioning the data.
            arguments
                XRaw
                Y
                args.purpose        char ...
                    {mustBeMember( args.purpose, ...
                    {'Creation', 'ForSubset'} )} = 'Creation'
                args.normalization  char ...
                    {mustBeMember( args.normalization, ...
                    {'PAD', 'LTN'} )} = 'LTN'
                args.normalizedPts  double ...
                    {mustBeNumeric, mustBePositive, mustBeInteger} = 101
                args.normalizeInput logical = false;
                args.padding        struct ...
                    {mustBeValidPadding}
                args.fda            struct ...
                    {mustBeValidFdParams}
                args.resampleRate   double ...
                    {mustBeNumeric} = 1
                args.hasDerivative  logical = false

            end

            self.XRaw = XRaw;
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
            self.resampleRate = args.resampleRate;
            self.hasDerivative = args.hasDerivative;

            % create smooth functions for the data
            [XFd, self.XDim] = smoothRawData( XRaw, ...
                                            self.padding, self.fda );

            % re-create cell array of time series after smoothing
            % resampling, as required
            tSpanResampled = linspace( ...
                self.fda.tSpan(1), self.fda.tSpan(end), ...
                fix( self.padding.length/self.resampleRate )+1 );
            
            XEval = eval_fd( tSpanResampled, XFd );
            
            if self.hasDerivative
                % include the first derivative as further channels
                DXEval = eval_fd( tSpanResampled, XFd, 1 );
                XEval = cat( 3, XEval, DXEval );
            end

            self.nObs = size( XEval, 2 );
            self.XChannels = size( XEval, 3 );
            
            % adjust lengths
            self.XDim = ceil( size( XEval, 1 )*self.XDim/self.padding.length );
            self.padding.length = max( self.XDim );
            
            % recreate the cell time series
            self.X = extractXSeries( XEval, self.XDim, ...
                            self.padding.length, self.padding.location );
            
            % prepare normalized data of fixed length
            self.XN = normalizeXSeries( self.X, self.normalizedPts, ...
                                        self.normalization, self.padding );
            self.XNDim = size( self.XN, 1 );

            if self.normalizeInput
                % also apply it to the input
                self.X = self.XN;
            end

            % assign category labels
            self.YLabels = categorical( unique(self.Y) );
            self.YDim = length( self.YLabels );

            % generate FDA objects from FDA parameters
            self.fda.basisFd = create_bspline_basis( ...
                            [ self.fda.tSpan(1), self.fda.tSpan(end) ], ...
                              self.fda.nBasis, self.fda.basisOrder);

            self.fda.fdPar = fdPar( self.fda.basisFd, ...
                                    self.fda.penaltyOrder, ...
                                    self.fda.lambda );
            
            % re-profile the time span
            self.fda.tSpan = linspace( self.fda.tSpan(1), ...
                                       self.fda.tSpan(end), ...
                                       self.normalizedPts );

        end


        function thisSubset = partition( self, idx )
            % Create the subset of this modelDataset
            % using the indices specified
            arguments
                self        modelDataset
                idx         logical 
            end

            subXRaw = split( self.XRaw, idx );
            subY = self.Y( idx );

            thisSubset = modelDataset( subXRaw, subY, ...
                                       purpose='ForSubset' );

            thisSubset.normalization = self.normalization;
            thisSubset.normalizedPts = self.normalizedPts;
            thisSubset.padding = self.padding;
            thisSubset.fda = self.fda;
            thisSubset.resampleRate = self.resampleRate;
            thisSubset.hasDerivative = self.hasDerivative;

            thisSubset.X = split( self.X, idx );
            thisSubset.XN = split( self.XN, idx );
            thisSubset.XDim = self.XDim( idx );

            thisSubset.XNDim = self.XNDim;
            thisSubset.XChannels = self.XChannels;
            thisSubset.YDim = self.YDim;
            thisSubset.YLabels = self.YLabels;
            thisSubset.nObs = sum( idx );

        end


        function isFixed = isFixedLength( self )
            % return whether data is time-normalized
            arguments
                self    modelDataset
            end

            isFixed = strcmp( self.normalization, 'LTN' );
            
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
                    {'return', 'discard'} )} = 'return'
            end

            [ ds, XNfmt ] = createDatastore( self.X, self.XN, self.Y );

            % setup the minibatch queues
            if iscell( self.X )
                
                % setup the minibatch preprocessing function
                preproc = @( X, XN, Y ) preprocMiniBatch( X, XN, Y, ...
                              self.padding.value, ...
                              self.padding.location );

                mbq = minibatchqueue(  ds,...
                                  'MiniBatchSize', batchSize, ...
                                  'PartialMiniBatch', args.partialBatch, ...
                                  'MiniBatchFcn', preproc, ...
                                  'MiniBatchFormat', {'CTB', XNfmt, 'CB'} );
            else
                % no preprocessing required

                mbq = minibatchqueue( ds,...
                                  'MiniBatchSize', batchSize, ...
                                  'PartialMiniBatch', args.partialBatch, ...
                                  'MiniBatchFormat', {'CB', XNfmt, 'BC'} );

            end

        end


    end

end


function [XFd, XLen] = smoothRawData( X, padding, fda )

    % find the series lengths (capped at padLen)
    XLen = min( cellfun( @length, X ), padding.length );

    % pad the series for smoothing
    X = padData( X, padding.length, padding.value, padding.location );
    
    % create a basis for smoothing with a knot at each point
    basisFd = create_bspline_basis( [fda.tSpan(1) fda.tSpan(end)], ...
                                       fda.nBasis, ...
                                       fda.basisOrder );
    % setup the smoothing parameters
    fdParams = fdPar( basisFd, ...
                      fda.penaltyOrder, ...
                      fda.lambda );

    % create the smooth functions
    XFd = smooth_basis( fda.tSpan, X, fdParams );

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
