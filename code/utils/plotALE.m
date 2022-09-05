function plotALE( thisModel, args )
    % Update the Accumulated Local Effects plot
    arguments
        thisModel           {mustBeA( thisModel, ...
            { 'FullRepresentationModel', ...
              'SubRepresentationModel' })}
        args.quantiles      double = []
        args.pts            double = []
        args.type           char ...
            {mustBeMember( args.type, {'Network', 'Model'})} = 'Network'
        args.axis           = []
    end

    switch args.type
        case 'Network'
            A = thisModel.AuxNetworkALE;
            axis = thisModel.Axes.AuxNetwork;
            name = 'Auxiliary Network';
        case 'Model'
            A = thisModel.AuxModelALE;
            axis = thisModel.Axes.AuxModel;
            name = 'Auxiliary Model';
    end

    if ~isempty(args.pts)
        A = args.pts;
    end

    if isempty(args.quantiles)
        Q = thisModel.ALEQuantiles;
    else
        Q = args.quantiles;
    end

    if ~isempty(args.axis)
        axis = args.axis;
    end

    nCodes = size( A, 1 );
    hold( axis, 'off');
    for i = 1:nCodes

        plot( axis, Q, A(i,:), 'LineWidth', 1 );
        hold( axis, 'on' );

    end
    
    hold( axis, 'off');
    
    title( axis, name );
    xlim( axis, [0,1] );
    xlabel( axis, 'Quantiles' );
    ylabel( axis, 'ALE' );   
    axis.YAxis.TickLabelFormat = '%.2f';
    
    finalisePlot( axis, square = true );
    
end

