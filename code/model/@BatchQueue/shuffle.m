function shuffle( self )
    % Shuffle the batches for a new epoch
    arguments
        self        BatchQueue
    end

    self.Batches = randomBatches( self.NumObs, self.BatchSize );
    self.CurrentBatch = 0;

end