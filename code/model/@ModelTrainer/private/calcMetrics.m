function [ metric, dlZ ] = calcMetrics( thisModel, dlX )
    % Compute various supporting metrics
    arguments
        thisModel   SubAEModel
        dlX         dlarray
    end

    % generate full-set encoding
    dlZ = thisModel.encode( dlX, convert = false );

    % record latent codes correlation
    [ metric.ZCorrelation, metric.ZCovariance ] = ...
                    latentCodeCorrelation( dlZ, summary = true );

    % generate validation components
    XC = thisModel.calcLatentComponents( dlZ );

    % record latent codes correlation
    [ metric.XCCorrelation, metric.XCCovariance ] = ...
                        latentComponentCorrelation( XC, summary = true );
    metric.XCCorrelation = mean( metric.XCCorrelation );
    metric.XCCovariance = mean( metric.XCCovariance );

    % convert to table
    metric = struct2table( metric );

end