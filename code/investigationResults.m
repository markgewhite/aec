function results = investigationResults( names, path, ...
                                         parameters, values, setup, ...
                                         memorySaving, resume )

    thisInvestigation = Investigation( names, path, ...
                             parameters, values, setup, resume );

    thisInvestigation.saveDataPlot;

    thisInvestigation.conserveMemory( memorySaving );

    results = thisInvestigation.save;

end