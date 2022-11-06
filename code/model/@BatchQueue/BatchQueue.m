classdef BatchQueue < handle
    % Faster replacement for Matlab's minibatchqueue

    properties
        Data            % cell array holding all variable arrays
        IterDim         % iteration dimensions for each data array
        NumInputs       % number of variable inputs
        NumOutputs      % number of preprocessed outputs
        NumBatches      % number of batches produced per epoch
        NumObs          % total number observations
        CurrentBatch    % current batch number
        BatchSize       % number of observations in a batch
        Batches         % logical array defining batches
        BatchFcn        % preprocessing function handle
        BatchFormat     % dlarray labels for the outputs
        ResetOrder      % sorted order for reset
    end

    methods

        function self = BatchQueue( D, args )
            % Initialize the batch queue
            arguments (Repeating)
                D                   {mustBeA(D,{'cell', 'double', 'single'})}
            end
            arguments
                args.IterDim        double ...
                    {mustBeInteger, mustBePositive} = []
                args.BatchSize      double ...
                    {mustBeInteger, mustBePositive} = 100
                args.NumOutputs     double ...
                    {mustBeInteger, mustBePositive} = []
                args.BatchFcn       function_handle = []
                args.BatchFormat    string = []
                args.ResetOrder     double ...
                    {mustBeInteger, mustBePositive} = []
            end

            % store data
            self.NumInputs = length(D);
            self.Data = D;

            % check inputs and outputs
            if isempty(args.NumOutputs)
                self.NumOutputs = self.NumInputs;
            else
                self.NumOutputs = args.NumOutputs;
            end

            % check iteration dimensions
            if isempty(args.IterDim)
                self.IterDim = ones(1,self.NumInputs);
            elseif length(args.IterDim)==self.NumInputs
                self.IterDim = args.IterDim;
            else
                eid = 'BatchQueue:IterDim';
                msg = 'Number of iteration dimensions does not match number of data arrays.';
                throwAsCaller( MException(eid,msg) );
            end

            % check data array lengths
            n = size( self.Data{1}, self.IterDim(1) );
            for i = 2:self.NumInputs
                if size( self.Data{i}, self.IterDim(i) ) ~= n
                    eid = 'BatchQueue:NumObs';
                    msg = 'Data arrays do not all have the same number of observations.';
                    throwAsCaller( MException(eid,msg) );
                end
            end
            self.NumObs = n;

            % check batch formats
            if length(args.BatchFormat)~=self.NumOutputs
                eid = 'BatchQueue:BatchFormat';
                msg = 'Number of batch formats does not match expected outputs.';
                throwAsCaller( MException(eid,msg) );
            end
            self.BatchFormat = args.BatchFormat;
            self.BatchSize = args.BatchSize;
            self.BatchFcn = args.BatchFcn;

            % set the reset order
            if isempty(args.ResetOrder)
                self.ResetOrder = 1:self.NumObs;
            elseif length(args.ResetOrder)~=self.NumObs
                eid = 'BatchQueue:ResetOrderLength';
                msg = 'ResetOrder length does not match the number of observations.';
                throwAsCaller( MException(eid,msg) );
            elseif length(unique(args.ResetOrder))~=self.NumObs
                eid = 'BatchQueue:ResetOrderNonUnique';
                msg = 'ResetOrder has duplicate indices.';
                throwAsCaller( MException(eid,msg) );
            else
                self.ResetOrder = args.ResetOrder;
            end

            % create batches (data partitions)
            self.Batches = resetBatches( self.NumObs, self.BatchSize, self.ResetOrder );
            self.NumBatches = size( self.Batches, 2 );

            % reset the batch count
            self.CurrentBatch = 0;

        end

        % class methods
        shuffle( self );
        reset( self );
        varargout = next( self )
    
    end


end
