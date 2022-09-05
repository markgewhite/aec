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

        if args.Holdout > 0
            % holdout partitioning
            cvpart = cvpartition( length( uniqueUnit ), ...
                                      Holdout = args.Holdout );
            selection = ismember( unit, uniqueUnit( training(cvpart) ));
        else
            % no partitioning - select all
            selection = true( self.NumObs, 1 );
        end
      
    else
        % K-fold partitioning
        if args.KFold > 1

            cvpart = cvpartition( length( uniqueUnit ), ...
                                  KFold = args.KFold );
            
            if length( uniqueUnit ) <= length( unit )
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

        else
            % no partitioning - select all
            selection = true( self.NumObs, 1 );

        end

    end

end
