function mbq = getMiniBatchQueue( self, batchSize, XLabels, XNLabels, args )
    % Create a minibatch queue
    arguments
        self                ModelDataset
        batchSize           double ...
            {mustBeInteger, mustBePositive}
        XLabels             char
        XNLabels            char
        args.lengthOrder    logical = false
        args.partialBatch   char ...
            {mustBeMember( args.partialBatch, ...
            {'return', 'discard'} )} = 'discard'
    end

    if args.lengthOrder
        % sort in ascending order of length
        XLen = cellfun( @length, self.XInput );
        [ ~, orderIdx ] = sort( XLen, 'descend' );
    else
        orderIdx = [];
    end

    % setup the minibatch preprocessing function
    preproc = @( X, XN, P, Y, I ) preprocMiniBatch( X, XN, P, Y, I, ...
                                                    self.Padding.Value, ...
                                                    self.Padding.Location );

    I = 1:self.NumObs;
    mbq = BatchQueue(  self.XInput, self.XTarget, self.PInput, self.Y, I, ...
                       IterDim = [1, 2, 1, 1, 2], ...
                       NumOutputs = 4, ...
                       BatchSize = batchSize, ...
                       BatchFcn = preproc, ...
                       BatchFormat = {XLabels, XNLabels, 'CB', 'CB'}, ...
                       ResetOrder = orderIdx );

end