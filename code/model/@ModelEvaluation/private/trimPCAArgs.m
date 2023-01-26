function newArgs = trimPCAArgs( args )
    % limit the arguments for PCA models to relevant fields

    pcaFields = {'NumCompLines', 'ComponentType', 'ComponentCentering', ...
                 'KFolds', 'ZDim', 'AuxModelType', ...
                 'name', 'path', ...
                 'ShowPlots', 'CompressionLevel' };
    for i = 1:length(pcaFields)
        if isfield( args, pcaFields{i} )
            newArgs.(pcaFields{i}) = args.(pcaFields{i});
        end
    end

end