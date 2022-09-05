function self = plotAllALE( self, arg )
    % Plot all the ALE curves from the sub-models
    arguments
        self            FullRepresentationModel
        arg.type        char {mustBeMember( ...
            arg.type, {'Model', 'Network'} )} = 'Model'
    end

    % create a temporary figure
    fig = figure;
    fig.Visible = false;
    fig.Position(3) = 50 + self.KFolds*200;

    % create all subplots
    axes = gobjects( self.KFolds, 1 );
    for k = 1:self.KFolds
        axes( k ) = subplot( 1, self.KFolds, k );
    end

    % plot all the components across figures
    for k = 1:self.KFolds
        plotALE( self.SubModels{k}, type = arg.type, ...
                 axis = axes(k) );
    end

    % save the figures and then close
    name = strcat( self.Info.Name, 'AllModelALE-', arg.type );
    figALE.ALE = fig;
    savePlots( figALE, self.Info.Path, name );
    close( fig );

end
