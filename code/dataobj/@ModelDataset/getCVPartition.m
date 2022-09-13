function selection = getCVPartition( self, args )
    % Generate a CV partition for the dataset
    arguments
        self                ModelDataset
        args.Holdout        double ...
            {mustBeInRange(args.Holdout, 0, 1)}
        args.KFolds         double ...
            {mustBeInteger, mustBePositive}
        args.Repeats        double ...
            {mustBeInteger, mustBePositive} = 1
        args.Identical      logical = false
    end

    if ~isfield( args, 'Holdout' ) && ~isfield( args, 'KFolds' )
        eid = 'ModelDataset:PartitioningNotSpecified';
        msg = 'Partitioning scheme not specified.';
        throwAsCaller( MException(eid,msg) );
    end

    if isfield( args, 'Holdout' ) && isfield( args, 'KFolds' )
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
        if args.KFolds > 1

            selection = false( self.NumObs, args.KFolds, args.Repeats );
            for r = 1:args.Repeats

                if r==1 || ~args.Identical
                    cvpart = cvpartition( length( uniqueUnit ), ...
                                          KFold = args.KFolds );
                end
                
                if length( uniqueUnit ) <= length( unit )
                    % partitioning unit is a grouping variable
                    for k = 1:args.KFolds
                        if args.Identical
                            % special case - make all partitions the same
                            f = 1;
                        else
                            f = k;
                        end
                        selection( :, k, r ) = ismember( unit, ...
                                        uniqueUnit( training(cvpart,f) ));
                    end
                else
                    selection( :, :, r ) = training( cvpart );
                end

            end
            selection = reshape( selection, [], args.KFolds*args.Repeats );
            
        else
            % no partitioning - select all
            selection = true( self.NumObs, 1 );

        end

    end

end
