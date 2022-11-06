function reset( self )
    % Reset the batches to a predetermined order
    arguments
        self        BatchQueue
    end

    self.Batches = resetBatches( self.NumObs, self.BatchSize, self.ResetOrder );
    self.CurrentBatch = 0;

end