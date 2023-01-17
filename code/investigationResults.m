function results = investigationResults( names, path, ...
                                         parameters, values, setup, ...
                                         memorySaving, resume, catchErrors )

    thisInvestigation = Investigation( names, path, ...
                             parameters, values, setup, ...
                             resume, catchErrors );

    thisInvestigation.saveDataPlot;

    thisInvestigation.conserveMemory( memorySaving );

    results = thisInvestigation.save;

end