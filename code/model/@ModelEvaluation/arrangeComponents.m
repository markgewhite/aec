function self = arrangeComponents( self )
    % Find the optimal arrangement for the sub-model's components
    % by finding the best set of permutations
    arguments
          self        ModelEvaluation
    end

    aModel = self.Models{1};
    permOrderIdx = perms( 1:aModel.ZDim );
    lb = [ length(permOrderIdx) ones( 1, self.NumModels-1 ) ];
    ub = length(permOrderIdx)*ones( 1, self.NumModels );
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
    latentComp = zeros( aModel.XInputDim, aModel.NumCompLines, ...
                        aModel.ZDim, aModel.XChannels, self.NumModels );
    for k = 1:self.NumModels
        latentComp(:,:,:,:,k) = self.Models{k}.LatentComponents;
    end
    
    % setup the objective function
    objFcn = @(p) arrangementError( p, latentComp );
    
    % run the genetic algorithm optimization
    [ componentPerms, componentMSE ] = ...
                        ga( objFcn, self.NumModels, [], [], [], [], ...
                            lb, ub, [], 1:self.NumModels, options );

    % generate the order from list of permutations
    self.ComponentOrder = zeros( self.NumModels, aModel.ZDim );
    for k = 1:self.NumModels
        self.ComponentOrder( k, : ) = permOrderIdx( componentPerms(k), : );
    end
    self.ComponentDiffRMSE = sqrt( componentMSE );

end