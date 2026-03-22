function res = objectiveFunctionMT(params, freq, app_res)
    res_params = params(1:3);
    thick_params = params(4:6);
    
    compute_res = zeros(1, length(freq));
    for ii = 1:length(freq)
        compute_res(ii) = modelMT(res_params, thick_params, freq(ii));
    end
   
   res = sum((compute_res - app_res).^2);
end