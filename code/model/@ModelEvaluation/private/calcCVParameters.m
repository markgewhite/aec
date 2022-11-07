function aggrP = calcCVParameters( models, group, set )
    % Average a specified group of parameters from models
    arguments
        models          cell
        group           char ...
            {mustBeMember( group, {'Loss', 'Correlations', 'Timing'} )}
        set             char ...
            {mustBeMember( set, {'Training', 'Validation'} )}
    end

    nModels = length( models );
    fields = fieldnames( models{1}.(group).(set) );
    nFields = length( fields );

    for i = 1:nFields

        fldDim = size( models{1}.(group).(set).(fields{i}) );
        P = zeros( [nModels fldDim] );
        for k = 1:nModels
           P(k,:,:) = models{k}.(group).(set).(fields{i});
        end
        aggrP.Mean.(fields{i}) = squeeze(mean(P,1));
        aggrP.SD.(fields{i}) = squeeze(std(P,1));

    end

end