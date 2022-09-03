function plotALE( thisModel, Z, A, args )
    % Update the Z distributions plot
    arguments
        thisModel           {mustBeA( thisModel, ...
            { 'FullRepresentationModel', ...
              'CompactRepresentationModel' })}
        Z                   double
        A                   double
        args.type           char ...
            {mustBeMember( args.type, {'Network', 'Model'})} = 'Network'
    end

    switch args.type
        case 'Network'
            axis = thisModel.Axes.AuxNetwork;
            name = 'Auxiliary Network';
        case 'Model'
            axis = thisModel.Axes.AuxModel;
            name = 'Auxiliary Model';
    end

    nCodes = size( A, 1 );
    hold( axis, 'off');
    for i = 1:nCodes

        plot( axis, Z(i,:), A(i,:), 'LineWidth', 1 );
        hold( axis, 'on' );

    end
    
    hold( axis, 'off');
    
    title( axis, name );
    xlabel( axis, 'Z' );
    ylabel( axis, 'ALE(Z)' );   
    axis.YAxis.TickLabelFormat = '%.2f';
    
    finalisePlot( axis, square = true );
    
end

