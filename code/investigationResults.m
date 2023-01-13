function results = investigationResults( names, path, ...
                                         parameters, values, setup, ...
                                         memorySaving, resume )

    thisRun = Investigation( names, path, ...
                             parameters, values, setup, resume );

    results = thisRun.getResults;

    thisRun.conserveMemory( memorySaving );

    thisRun.save;

end