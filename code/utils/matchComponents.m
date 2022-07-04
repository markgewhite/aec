function componentRMSE = matchComponents( thisModel )
    % Match up the latent components of sub-models
    arguments
        thisModel   FullRepresentationModel
    end

    % get all the possible permutations of ordering the components
    permOrder = perms(1:thisModel.ZDim);

    nComparisons = thisModel.KFolds*(thisModel.KFolds-1)/2;
    nCompComparisons = length(permOrder);

    nLinesPerComponent = size( thisModel.LatentComponents, 2 )/thisModel.ZDim;

    componentRMSE = zeros( nCompComparisons, nCompComparisons );

    for k1 = 1:thisModel.KFolds
        thisK1Model = thisModel.SubModels{k1};

        for p = 1:length(permOrder)

            for k2 = k1+1:thisModel.KFolds
                thisK2Model = thisModel.SubModels{k2};
        
                for q = 1:length(permOrder)

                    for d = 1:thisModel.ZDim
                        c1A = (permOrder(p,d)-1)*nLinesPerComponent + 1;
                        c1B = c1A + nLinesPerComponent - 1;
    
                        c2A = (permOrder(q,d)-1)*nLinesPerComponent + 1;
                        c2B = c2A + nLinesPerComponent - 1;
    
                        compC1 = thisK1Model.LatentComponents(:,c1A:c1B);
                        compC2 = thisK2Model.LatentComponents(:,c2A:c2B);
                                                
                        componentRMSE(p,q) = componentRMSE(p,q) + ...
                                        mean( (compC1 - compC2).^2, 'all' );
                    end
                    
                end
            end

        end
    end

end