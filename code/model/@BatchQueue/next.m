function varargout = next( self )
    % Generate the next batch
    arguments
        self        BatchQueue
    end

    % update the batch counter
    self.CurrentBatch = self.CurrentBatch+1;
    if self.CurrentBatch>self.NumBatches
        return
    end

    % obtain the batch using dynamic indexing
    data = cell( self.NumInputs, 1 );
    for i = 1:self.NumInputs
        % create dimension index array with right number of dimensions
        dimIdx = repmat( {':'}, 1, ndims(self.Data{i}));
        % set the filter for the iteration dimension
        dimIdx{self.IterDim(i)} = self.Batches(:,self.CurrentBatch);
        % perform the subselection
        data{i} = subsref( self.Data{i}, substruct('()',dimIdx));
    end

    % call the preprocessing function
    batch = cell( 1, self.NumOutputs );
    [batch{:}] = self.BatchFcn( data{:} );

    % create the dlarrays with the required labelling
    varargout = cell( 1, self.NumOutputs );
    for i = 1:self.NumOutputs
        varargout{i} = dlarray( batch{i}, self.BatchFormat(i) );
    end
    

end