function [model, ZTrnMean, ZTrnSD] = trainAuxModel( modelType, dlZTrn, dlYTrn )
    % Train a non-network auxiliary model
    arguments
        modelType   string ...
            {mustBeMember(modelType, {'Logistic', 'Fisher', 'SVM', 'LR'} )}
        dlZTrn      dlarray
        dlYTrn      dlarray
    end
    
    % convert to double for models which don't take dlarrays
    ZTrn = double(extractdata( gather(dlZTrn) ))';
    YTrn = double(extractdata( gather(dlYTrn) ));

    % standardize
    ZTrnMean = mean( ZTrn );
    ZTrnSD = std( ZTrn );
    ZTrn = (ZTrn-ZTrnMean)./ZTrnSD;
    
    % fit the appropriate model
    switch modelType
        case 'Logistic'
            model = fitclinear( ZTrn, YTrn, Learner = "logistic" );
        case 'Fisher'
            model = fitcdiscr( ZTrn, YTrn );
        case 'SVM'
            model = fitcecoc( ZTrn, YTrn );
        case 'LR'
            model = fitrlinear( ZTrn, YTrn );
    end

end