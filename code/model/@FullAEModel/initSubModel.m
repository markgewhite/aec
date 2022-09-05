function self = initSubModel( self, k )
    % Initialize a sub-model
    arguments
        self            FullAEModel
        k               double
    end

    self.SubModels{k} = CompactAEModel( self, k );
    if self.IdenticalNetInit && k==1
        self.InitializedNets = self.SubModels{k}.Nets;
    end

end