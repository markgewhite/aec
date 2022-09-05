function self = setLatentComponents( self )
    % Calculate the cross-validated latent components
    % by averaging across the sub-models
    arguments
        self        FullRepresentationModel
    end

    XC = self.SubModels{1}.LatentComponents;
    for k = 2:self.KFolds

        if isempty( self.ComponentOrder )
            % use model arrangement
            comp = self.SubModels{k}.LatentComponents;
        else
            % use optimized arrangement
            comp = reshape( self.SubModels{k}.LatentComponents, ...
                        self.XInputDim, self.NumCompLines, [] );
            comp = comp( :, :, self.ComponentOrder(k,:) );
            comp = reshape( comp, self.XInputDim, [] );
        end

        XC = XC + comp;
    
    end

    self.CVLatentComponents = XC/self.KFolds;

end