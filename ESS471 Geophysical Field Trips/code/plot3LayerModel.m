function plot3LayerModel(resistivities, thicknesses)
    % Get current axes
    ax = gca;
    hold(ax, 'on');
    grid(ax, 'on');
    box(ax, 'on');
    
    % Set Y-axis direction (depth increases downward)
    set(ax, 'YDir', 'reverse');
    
    % Define layer heights (second layer thicker than the first, third extends to bottom)
    layer_heights = [0.15, 0.25, 0.6];  % Layer 1: 0.15, Layer 2: 0.25, Layer 3: 0.6
    
    % Calculate starting positions of each layer (to ensure continuity)
    layer_starts = [0, cumsum(layer_heights(1:2))];
    
    % Define colors for each layer
    colors = [0.9 0.95 1;      % Layer 1 (light blue)
              0.8 0.9 1;       % Layer 2 (medium blue)
              0.7 0.85 1];     % Layer 3 (dark blue)
    
    % Draw each layer
    for i = 1:3
        % Calculate position and height of current layer
        rect_y = layer_starts(i);
        rect_height = layer_heights(i);
        
        % Draw rectangle representing the layer
        rectangle(ax, 'Position', [0.1, rect_y, 0.8, rect_height], ...
                  'FaceColor', colors(i, :), ...
                  'EdgeColor', 'k', ...
                  'LineWidth', 1.5);
        
        % Add resistivity label
        text(ax, 0.5, rect_y + rect_height/2, ...
             sprintf('\\rho_{%d} = %.1f \\Omega\\cdotm', i, resistivities(i)), ...
             'FontSize', 12, 'FontWeight', 'bold', ...
             'HorizontalAlignment', 'center');
        
        % Add thickness label (only for the first two layers)
        if i < 3
            % Compute label position (just above the bottom of current layer)
            h_label_y = rect_y + rect_height - 0.02;
            
            % Draw thickness indicator line (only for the current layer)
            line(ax, [0.05, 0.05], [rect_y, rect_y + rect_height], ...
                 'Color', 'r', 'LineWidth', 2);
            
            % Add thickness text (left of the indicator line)
            text(ax, 0.02, rect_y + rect_height/2, ...
                 sprintf('h_{%d} = %.1f m', i, thicknesses(i)), ...
                 'FontSize', 10, 'Color', 'r', ...
                 'Rotation', 90, ...
                 'VerticalAlignment', 'middle', ...
                 'HorizontalAlignment', 'center');
        end

        % Add dashed horizontal boundary line (for layers below the first)
        if i > 1
            line(ax, [0, 1], [rect_y, rect_y], ...
                 'Color', 'k', 'LineWidth', 2, 'LineStyle', '--');
        end
    end
    
    % Draw top surface line
    line(ax, [0, 1], [0, 0], 'Color', 'k', 'LineWidth', 3);
    
    % Label the half-space at the top
    text(ax, 0.5, 0.95, 'half-space', ...
         'FontSize', 11, 'Color', [0.5 0.5 0.5], ...
         'HorizontalAlignment', 'center');
    
    % Set plot limits and formatting
    xlim(ax, [0, 1]);
    ylim(ax, [0, 1]);
    set(ax, 'XTick', []);
    set(ax, 'YTick', []);
    title(ax, '3-Layered Model','FontSize',20);
    ylabel(ax, 'Depth','FontSize',20);
    hold(ax, 'off');
end
