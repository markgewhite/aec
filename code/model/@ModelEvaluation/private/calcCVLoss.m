function cvLoss = calcCVLoss( models, set )
    % Calculate the aggregate cross-validated losses across all submodels
    % drawing on the pre-computed predictions 
    arguments
        models          cell
        set             char ...
            {mustBeMember( set, {'Training', 'Testing'} )}
    end

    nModels = length( models );

    pairs = [   {'XTarget', 'XHat'}; ...
                {'XInput', 'XHatSmoothed'}; ...
                {'Y', 'AuxModelYHat'}; ];
    fieldsToPermute = { 'XInput', 'XTarget', 'XHat', 'XHatSmoothed' };
    fieldsForAuxLoss = { 'AuxModelYHat', 'AuxNetworkYHat', 'ComparatorYHat' };

    % check if fields are present
    fields = unique( pairs );
    nPairs = length( pairs );
    for i = 1:nPairs
        if ~isfield( models{1}.Predictions.(set), fields{i} )
            pairs{i,:} = [];
        end
    end
    fields = unique( pairs );
    nPairs = length( pairs );

    % aggregate all the predictions for each field into one array
    for i = 1:length(fields)

        aggr.(fields{i}) = [];
        doPermute = ismember( fields{i}, fieldsToPermute );

        for k = 1:nModels

            data = models{k}.Predictions.(set).(fields{i});
            if doPermute
                data = permute( data, [2 1 3] );
            end
            aggr.(fields{i}) = [ aggr.(fields{i}); data ];

        end

    end

    scale = mean( cell2mat(cellfun( ...
                    @(m) m.Scale, models, UniformOutput = false )), 2 );

    for i = 1:nPairs

        A = aggr.(pairs{i,1});
        AHat = aggr.(pairs{i,2});

        if ismember( pairs{i,2}, fieldsForAuxLoss )
            % cross entropy loss
            cvLoss.(pairs{i,2}) = evaluateClassifier( A, AHat );

        else
            % mean squared error loss
            cvLoss.(pairs{i,2}) = reconLoss( A, AHat, scale );

            % permute dimensions for temporal losses
            A = permute( A, [2 1 3] );
            AHat = permute( AHat, [2 1 3] );

            % temporal mean squared error loss
            cvLoss.([pairs{i,2} 'TemporalMSE']) = ...
                                    reconTemporalLoss( A, AHat, scale );

            % temporal bias
            cvLoss.([pairs{i,2} 'TemporalBias']) = ...
                                    reconTemporalBias( A, AHat, scale );

            % temporal variance rearranging formula: MSE = Bias^2 + Var
            cvLoss.([pairs{i,2} 'TemporalVar']) = ...
                cvLoss.([pairs{i,2} 'TemporalMSE']) ...
                                - cvLoss.([pairs{i,2} 'TemporalBias']).^2;
        
        end
    
    end



end
