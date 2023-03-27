function closeFigures( self )
    % Close all figures
    arguments
        self            RepresentationModel
    end

    flds = fieldnames( self.Figs );
    for i = 1:length(flds)
        try
            close( self.Figs.(flds{i}) );
        catch
            disp(['Could not close figure = ' char(flds{i})]);
        end
    end

end