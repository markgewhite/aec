function genPaperCompPlots( path, title, names, nDims, nReports, nModels ) 
    % Compile component plots for the paper, re-formatting saved figures
    arguments
        path        mustBeFolder
        title       string
        names       strings
        nDims       double {mustBeInteger, mustBePositive}
        nReports    double {mustBeInteger, mustBePositive}
        nModels     double {mustBeInteger, mustBePositive}
    end

    for d = 1:nDims
        for i = 1:nReports
            for j = 1:nModels
                figpath = strcat( path, "/", names(i), "/", ...
                    names(i), "-Eval(", num2str(j), ",", num2str(d), ")/comp" );
            
                figfilename = strcat( names(i), "-Fold01.fig" );
                fig = openfig( fullfile(figpath, figfilename) );
                filename = strcat( title, "-", names(i), "-Comp(", ...
                    num2str(j), ",", num2str(d), ").pdf" );
    
                fig = formatIEEEFig( fig, ...
                                     width = "Bespoke", ...
                                     inches = d*1.5, ...
                                     size = "Medium", ...
                                     keepAxisLabels = false, ...
                                     keepAxisTicks = true, ...
                                     keepTitle = true, ...
                                     filename = filename );
    
    
            end
        end
    end

end