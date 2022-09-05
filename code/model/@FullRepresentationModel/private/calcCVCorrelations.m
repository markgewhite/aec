function aggrCorr = calcCVCorrelations( subModels, set )
    % Average correlations from submodels
    arguments
        subModels       cell
        set             char ...
            {mustBeMember( set, {'Training', 'Validation'} )}
    end

    nModels = length( subModels );
    fields = fieldnames( subModels{1}.Correlations.(set) );
    nFields = length( fields );

    for i = 1:nFields

        fldDim = size( subModels{1}.Correlations.(set).(fields{i}) );
        R = zeros( fldDim );
        for k = 1:nModels
           R = R + subModels{k}.Correlations.(set).(fields{i});
        end
        aggrCorr.(fields{i}) = R/nModels;

    end

end