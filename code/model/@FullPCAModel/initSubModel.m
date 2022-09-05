function self = initSubModel( self, k )
    % Initialize a sub-model
    arguments
        self            FullPCAModel
        k               double
    end

    self.SubModels{k} = SubPCAModel( self, k );

end