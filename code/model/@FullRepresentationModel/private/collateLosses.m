function aggrLoss = collateLosses( subModels, set )
    % Collate the losses from the submodels
    arguments
        subModels       cell
        set             char ...
            {mustBeMember( set, {'Training', 'Validation'} )}
    end

    nModels = length( subModels );
    fields = fieldnames( subModels{1}.Loss.(set) );
    nFields = length( fields );

    for i = 1:nFields

        fldDim = size( subModels{1}.Loss.(set).(fields{i}) );
        thisAggrLoss = zeros( [nModels fldDim] );
        for k = 1:nModels
           thisAggrLoss(k,:,:) = subModels{k}.Loss.(set).(fields{i});
        end
        aggrLoss.(fields{i}) = thisAggrLoss;

    end

end