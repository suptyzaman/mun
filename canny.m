% Canny Edge Detection without Built-in Functions (Indexing starts from 1)
% Read the input image
input_image = imread('C:/Users/USER/Downloads/gg.jpg'); % Replace 'Pic.png' with your image file
input_image = rgb2gray(input_image); % Convert to grayscale if it's a color image
input_image = double(input_image); % Convert to double for calculations

% Step 1: Noise Reduction using Gaussian filter
sigma = 1.0; % Standard deviation for Gaussian filter
kernel_size = 5; % Kernel size (must be odd)
Gaussian_filter = fspecial('gaussian', kernel_size, sigma);
smoothed_image = conv2(input_image, Gaussian_filter, 'same');

% Step 2: Compute Gradients using Sobel Operators
Gx = [-1 0 1; -2 0 2; -1 0 1]; % Sobel kernel in X direction
Gy = [-1 -2 -1; 0 0 0; 1 2 1]; % Sobel kernel in Y direction

% Apply Sobel operators to compute the gradients
gradient_x = conv2(smoothed_image, Gx, 'same');
gradient_y = conv2(smoothed_image, Gy, 'same');

% Compute gradient magnitude and direction
gradient_magnitude = sqrt(gradient_x.^2 + gradient_y.^2);
gradient_direction = atan2(gradient_y, gradient_x); % Gradient direction

% Step 3: Non-Maximum Suppression
[rows, cols] = size(gradient_magnitude);
suppressed_image = zeros(rows, cols);

for i = 2:rows-1 % Loop starts from 2 and ends at rows-1 (skip boundaries)
    for j = 2:cols-1 % Loop starts from 2 and ends at cols-1 (skip boundaries)
        angle = gradient_direction(i,j) * 180 / pi; % Convert angle to degrees
        angle = mod(angle + 180, 180); % Ensure the angle is between 0 and 180
        
        % Interpolate between neighboring pixels based on gradient direction
        if (0 <= angle < 22.5) || (157.5 <= angle <= 180)
            neighbor1 = gradient_magnitude(i,j-1); % Left
            neighbor2 = gradient_magnitude(i, j+1); % Right
        elseif (22.5 <= angle < 67.5)
            neighbor1 = gradient_magnitude(i-1, j+1); % Top-right
            neighbor2 = gradient_magnitude(i+1, j-1); % Bottom-left
        elseif (67.5 <= angle < 112.5)
            neighbor1 = gradient_magnitude(i-1, j); % Top
            neighbor2 = gradient_magnitude(i+1, j); % Bottom
        else
            neighbor1 = gradient_magnitude(i-1, j-1); % Top-left
            neighbor2 = gradient_magnitude(i+1, j+1); % Bottom-right
        end
        
        % Non-Maximum Suppression
        if (gradient_magnitude(i,j) >= neighbor1) && (gradient_magnitude(i,j) >= neighbor2)
            suppressed_image(i,j) = gradient_magnitude(i,j);
        else
            suppressed_image(i,j) = 0;
        end
    end
end

% Step 4: Edge Tracking by Hysteresis
low_threshold = 0.05 * max(suppressed_image(:)); % Low threshold
high_threshold = 0.15 * max(suppressed_image(:)); % High threshold

edges = zeros(rows, cols);

for i = 2:rows-1 % Loop starts from 2 and ends at rows-1 (skip boundaries)
    for j = 2:cols-1 % Loop starts from 2 and ends at cols-1 (skip boundaries)
        if suppressed_image(i,j) > high_threshold
            edges(i,j) = 1; % Strong edge
        elseif suppressed_image(i,j) > low_threshold
            % Check for weak edge connected to a strong edge
            if any(any(suppressed_image(i-1:i+1, j-1:j+1) > high_threshold))
                edges(i,j) = 1; % Weak edge connected to strong edge
            else
                edges(i,j) = 0; % Not an edge
            end
        else
            edges(i,j) = 0; % No edge
        end
    end
end

% Display the results
figure;
subplot(2,1,1); imshow(uint8(input_image)); title('Original Image');

subplot(2,1, 2); imshow(uint8(edges * 255)); title('Final Edges (Canny)');
