function XC = calcCVComponents( self )
    % Calculate the cross-validated latent components
    % by averaging across the models
    arguments
        self        ModelEvaluation
    end

    XC = self.Models{1}.LatentComponents;
    for k = 2:self.KFolds

        if isempty( self.ComponentOrder )
            % use model arrangement
            comp = self.Models{k}.LatentComponents;
        else
            % use optimized arrangement
            comp = self.Models{k}.LatentComponents( :,:,self.ComponentOrder(k,:),:,: );
        end

        XC = XC + comp;
    
    end

    XC = XC/self.KFolds;

end