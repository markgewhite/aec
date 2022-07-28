function T0 = genPaperResultsTable( results, fields, groupSizes )
    % Generate a results table for the paper
    arguments
        results         cell
        fields          string
        groupSizes      double
    end

    nReports = length( results );
    nFields = length( fields );
    allFields = fieldnames( results{1}.TestingResults  );
    [nModels, nDatasets] = size( results{1}.TestingResults.(allFields{1}) );
    
    fieldNames = strings( nFields, nModels );
    groupings = cell( nFields, 1 );
    
    % initialize results table
    for i = 1:nFields
        for j = 1:nModels
            fieldNames(i,j) = strcat( fields(i), num2str(j) );
            T.Median.(fieldNames(i,j)) = zeros( nReports, 1 );
            if nDatasets > 1
                T.IQR.(fieldNames(i,j)) = zeros( nReports, 1 );
                if nModels > 1
                    T.Friedman.(fieldNames(i,j)) = zeros( nReports, 1 );
                end
            end
        end
        groupings{i} = (i-1)*nModels+(1:groupSizes(i));
    end
    
    % compile results into tables
    for i = 1:nFields
    
        for k = 1:nReports
    
            q = zeros( nDatasets, nModels );
            for j = 1:nModels
    
                for m = 1:nDatasets
                    q(m,j) = results{k}.TestingResults.(fields(i))(j,m);
                end
    
                T.Median.(fieldNames(i,j))(k) = median( q(:,j) );
                if nDatasets > 1
                    T.IQR.(fieldNames(i,j))(k) = iqr( q(:,j) );
                end

            end
    
            if nDatasets > 1 && nModels > 1
                % conduct Friedman's ANOVA comparison between models
                [ ~, ~, stats ] = friedman( q(1:groupSizes(i),:), 1, "off" );
                if groupSizes(i)==nModels
                    controlGrp = nModels; % PCA
                else
                    controlGrp = 1; % FC
                end
                ci = multcompare( stats, ...
                                  CriticalValueType = "dunnett", ...
                                  ControlGroup = controlGrp, ...
                                  Display= "off" );
                
                m = 0;
                for j = setdiff( 1:nModels, controlGrp )
                    m = m+1;
                    T.Friedman.(fieldNames(i,j))(k) = ci(m, 6);
                end
            end

        end
    
    
    
    end
    T0 = struct2table( T.Median );

    if nModels > 1
        embolden = "Rows";
    else
        embolden = "None";
    end

    if nDatasets > 1
        T1 = struct2table( T.IQR );
        if nModels > 1
            T2 = struct2table( T.Friedman );
        else
            T2 = [];
        end
        
        T0 = genPaperTableCSV( T0, T1, T2, ...
                               direction = embolden, criterion = "Smallest", ...
                               groups = groupings );
    else
        T0 = genPaperTableCSV( T0, [], [], ...
                               direction = embolden, criterion = "Smallest", ...
                               groups = groupings );
    end

end