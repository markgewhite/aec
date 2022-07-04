function componentOrder = matchComponents( thisModel )
    % Match up the latent components of sub-models
    arguments
        thisModel   FullRepresentationModel
    end

    % get all the possible permutations of ordering the components
    p = perms(1:thisModel.ZDim);

    nComparisons = thisModel.KFolds*(thisModel.KFolds-1)/2;
    nCompComparisons = length(p);

    nLinesPerComponent = size( thisModel.LatentComponents, 2 )/thisModel.ZDim;

    componentRMSE = zeros( nComparisons, nCompComparisons );
    rankingOrder = zeros( nCompComparisons, nComparisons );
    componentOrder = zeros( nCompComparisons, thisModel.ZDim, nComparisons );

    i = 0;
    for k1 = 1:thisModel.KFolds
        thisK1Model = thisModel.SubModels{k1};
        
        for k2 = k1+1:thisModel.KFolds
            thisK2Model = thisModel.SubModels{k2};

            i = i+1;
            for j = 1:length(p)
                    
                for d = 1:thisModel.ZDim
                    c1A = (d-1)*nLinesPerComponent + 1;
                    c1B = c1A + nLinesPerComponent - 1;

                    c2A = (p(j,d)-1)*nLinesPerComponent + 1;
                    c2B = c2A + nLinesPerComponent - 1;

                    compC1 = thisK1Model.LatentComponents(:,c1A:c1B);
                    compC2 = thisK2Model.LatentComponents(:,c2A:c2B);
                                            
                    componentRMSE(i,j) = componentRMSE(i,j) + ...
                                    mean( (compC1 - compC2).^2, 'all' );
                end
                componentRMSE(i,j) = componentRMSE(i,j)/thisModel.ZDim;

            end
            [~, rankingOrder(:,i)] = sort( componentRMSE(i,:) );
            componentOrder(:,:,i) = p( rankingOrder(:,i), : );

        end
    end

end