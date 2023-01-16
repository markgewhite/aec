function genPaperCompPlots( path, title, names, nDims, nReports, nModels ) 
    % Compile component plots for the paper, re-formatting saved figures
    arguments
        path        string
        title       string
        names       string
        nDims       double {mustBeInteger, mustBePositive}
        nReports    double {mustBeInteger, mustBePositive}
        nModels     double {mustBeInteger, mustBePositive}
    end

    for i = 1:nReports
        for d = 1:nDims
            for j = 1:nModels

                if not(nDims==1 || nModels==1)
                    evalCode = strcat(num2str(j), ",", num2str(d));
                elseif nDims==1
                    evalCode = num2str(j);
                else
                    evalCode = num2str(d);
                end

                figpath = strcat( path, "/", names(i), "/", ...
                    names(i), "-Eval(", evalCode, ")/comp" );
            
                figfilename = strcat( names(i), "(1).fig" );
                fig = openfig( fullfile(figpath, figfilename) );
                filename = strcat( title, "-", names(i), "-Comp(", ...
                    num2str(j), ",", num2str(d), ").pdf" );
    
                fig = formatIEEEFig( fig, ...
                                     width = "Components", ...
                                     keepXAxisLabels = false, ...
                                     keepYAxisLabels = false, ...
                                     filename = filename );
    
    
            end
        end
    end

end