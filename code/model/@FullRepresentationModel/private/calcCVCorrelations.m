function aggrCorr = calcCVCorrelations( models, set )
    % Average correlations from submodels
    arguments
        models          cell
        set             char ...
            {mustBeMember( set, {'Training', 'Validation'} )}
    end

    nModels = length( models );
    fields = fieldnames( models{1}.Correlations.(set) );
    nFields = length( fields );

    for i = 1:nFields

        fldDim = size( models{1}.Correlations.(set).(fields{i}) );
        R = zeros( fldDim );
        for k = 1:nModels
           R = R + models{k}.Correlations.(set).(fields{i});
        end
        aggrCorr.(fields{i}) = R/nModels;

    end

end