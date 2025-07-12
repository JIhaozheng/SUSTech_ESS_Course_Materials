function error = computeError(x, freq, app_res)
    res_params = x(1:3);
    thick_params = x(4:5);
    if any(res_params <= 0) || any(thick_params <= 0)
        error = inf; 
        return;
    end
    compute_res = zeros(size(freq));
    for i = 1:length(freq)
        compute_res(i) = modelMT(res_params, thick_params, freq(i));
    end
    error = sum((compute_res - app_res).^2);
end