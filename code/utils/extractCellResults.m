function [y, newNames] = extractCellResults( yCell, nCoeff, rowNames )
    % Extract cell array by extending the first dimension
    % It is assumed yCell pertains to the model coefficients
    arguments
        yCell       cell
        nCoeff      double {mustBeInteger, mustBePositive}
        rowNames    string = []
    end

    [nModels, nValues, nDatasets] = size( yCell );
    
    y = zeros( nModels*nCoeff, nValues, nDatasets );
    newNames = strings( 1, nModels*nCoeff );

    % pre-sort the coefficients with largest in magnitude first
    yCell = cellfun( @(x) sort( abs(x), 'descend'), yCell, ...
                     UniformOutput = false );

    % extract the y cell array into a longer numeric array
    for i = 1:nCoeff
        for j = 1:nModels
            for k = 1:nValues
                for l = 1:nDatasets
                    if i<=length( yCell{j,k,l} )
                        y((j-1)*nCoeff+i,k,l) = yCell{j,k,l}(i);
                    else
                        y((j-1)*nCoeff+i,k,l) = NaN;
                    end
                end
            end
        end
    end

    % update rowNames, if required
    if ~isempty( rowNames )
        for i = 1:nCoeff
            for j = 1:nModels
                newNames((j-1)*nCoeff+i) = strcat( rowNames(j), ...
                                               ['-' num2str(i)] );
            end
        end
    else
        newNames = [];
    end

end