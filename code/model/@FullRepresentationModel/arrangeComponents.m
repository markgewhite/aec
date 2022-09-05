function self = arrangeComponents( self )
    % Find the optimal arrangement for the sub-model's components
    % by finding the best set of permutations
    arguments
                  self        FullRepresentationModel
    end

    permOrderIdx = perms( 1:self.ZDim );
    lb = [ length(permOrderIdx) ones( 1, self.KFolds-1 ) ];
    ub = length(permOrderIdx)*ones( 1, self.KFolds );
    options = optimoptions( 'ga', ...
                            'PopulationSize', 400, ...
                            'EliteCount', 80, ...
                            'MaxGenerations', 300, ...
                            'MaxStallGenerations', 150, ...
                            'FunctionTolerance', 1E-6, ...
                            'UseVectorized', true, ...
                            'PlotFcn', {'gaplotbestf','gaplotdistance', ...
                                        'gaplotbestindiv' } );

    % pre-compile latent components across the sub-models for speed
    latentComp = zeros( self.XInputDim, self.NumCompLines, ...
                        self.ZDim, self.XChannels, self.KFolds );
    for k = 1:self.KFolds
        latentComp(:,:,:,:,k) = self.SubModels{k}.LatentComponents;
    end
    
    % setup the objective function
    objFcn = @(p) arrangementError( p, latentComp, self.ZDim );
    
    % run the genetic algorithm optimization
    [ componentPerms, componentMSE ] = ...
                        ga( objFcn, self.KFolds, [], [], [], [], ...
                            lb, ub, [], 1:self.KFolds, options );

    % generate the order from list of permutations
    self.ComponentOrder = zeros( self.KFolds, self.ZDim );
    for k = 1:self.KFolds
        self.ComponentOrder( k, : ) = permOrderIdx( componentPerms(k), : );
    end
    self.ComponentDiffRMSE = sqrt( componentMSE );

end