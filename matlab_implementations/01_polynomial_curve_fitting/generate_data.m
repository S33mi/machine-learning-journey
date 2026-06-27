function [x, t] = generate_data(N, noise_sigma)
    x = linspace(0, 1, N)';
    t = sin(2*pi*x) + noise_sigma * randn(N,1);
end
