function self = plotAllLatentComponents( self, arg )
    % Plot all the latent components from the sub-models
    arguments
        self                FullRepresentationModel
        arg.Rearranged      logical = false
    end

    figs = gobjects( self.KFolds, 1 );
    axes = gobjects( self.XChannels, self.ZDim, self.KFolds );
    for c = 1:self.XChannels

        % create a temporary large figure
        figs(c) = figure;
        figs(c).Visible = false;
        figs(c).Position(3) = 100 + self.ZDim*250;
        figs(c).Position(4) = 50 + self.KFolds*200;

        % create all subplots
        for k = 1:self.KFolds
            for d = 1:self.ZDim
                axes( c, d, k ) = ...
                    subplot( self.KFolds, self.ZDim, ...
                         (k-1)*self.ZDim + d );
            end
        end

    end

    % plot all the components across figures
    for k = 1:self.KFolds
        if arg.Rearranged
            plotLatentComp( self.SubModels{k}, ...
                        order = self.ComponentOrder(k,:), ...
                        axes = axes(:,:,k), ...
                        showTitle = (k==1), ...
                        showLegend = false, ...
                        showXAxis = (k==self.KFolds) );
        else
            plotLatentComp( self.SubModels{k}, ...
                        axes = axes(:,:,k), ...
                        showTitle = (k==1), ...
                        showLegend = false, ...
                        showXAxis = (k==self.KFolds) );
        end
    end

    % save the figures and then close
    name = strcat( self.Info.Name, 'AllKFolds' );
    for c = 1:self.XChannels
        figComp.Components = figs(c);
        savePlots( figComp, self.Info.Path, name );
        close( figs(c) );
    end

end   