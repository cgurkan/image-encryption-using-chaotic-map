
% Constants
initial_value = 0.815647;
r_param = 3.987;

% Test images
secret_img_name = 'lena_128.png';
cover_img_name  = 'sydney_512.png';

% images converted to grayscale
secret_image = imread(secret_img_name);
if size(secret_image, 3) == 3
    secret_image = rgb2gray(secret_image);  
end   


% Encrypt the image
%encrypted_image = chaotic_image_encryption(secret_image);


%test_arnold_mapping(img)
%test_counterlet(img)
enc_dec(secret_image)

function test_counterlet(img)    
    
    % Step 1
    pfilt = 'pkva';     
    dfilt = 'pkva';     
    nlevs = [0, 2, 3, 4] ;  
    coeff = pdfbdec(double(img), pfilt, dfilt, nlevs);  

    low_freq_subband = coeff{1};  

    % Step 2
    scrambled_low_freq = arnold_map_scramble(low_freq_subband);
    coeff_d = coeff;
    coeff_d{1} = scrambled_low_freq;  

    % Step 3
    scrambled_img = pdfbrec(coeff_d, pfilt, dfilt);   

    % Display results
    figure;
    subplot(1, 3, 1);
    imshow(img);
    title('Original Image');
    
    subplot(1, 3, 2);
    imshow(uint8(scrambled_img));
    title('Scrambled Image');    
    
    %=============================================
    
    % Step 1
    pfilt = 'pkva';     
    dfilt = 'pkva';     
    nlevs = [0, 2, 3, 4] ;  
    coeff_s = pdfbdec(double(scrambled_img), pfilt, dfilt, nlevs);

    scrambled_low_freq_s = coeff_s{1};  

    % Step 2
    unscrambled_low_freq = inverse_arnold_map_scramble(scrambled_low_freq_s);
    coeff_x = coeff_s;
    coeff_x{1} = double(unscrambled_low_freq); 

    % Step 3
    decrypted_image = pdfbrec(coeff_x, pfilt, dfilt);

    % Display results
    figure;
    subplot(1, 3, 1);
    imshow(scrambled_img);
    title('Scrambling Image');
    
    subplot(1, 3, 2);
    imshow(uint8(decrypted_image));
    title('Converted Image');   


    %Check scrambling differences
    mse = sum( sum( (decrypted_image - double(img)).^2 ) );
    mse = mse / prod(size(img));

    disp( sprintf('The mean square error is: %f', mse ) );
    disp(' ');
end

% Arnol Map Test
function test_arnold_mapping(img)
    arn_map= arnold_map_scramble(img);
    inv_arn_map= inverse_arnold_map_scramble(arn_map);
    
    % Display results
    figure;
    subplot(1, 3, 1);
    imshow(img);
    title('Original Image');
    
    subplot(1, 3, 2);
    imshow(uint8(arn_map));
    title('Encrypted Image');
    
    subplot(1, 3, 3);
    imshow(uint8(inv_arn_map));
    title('Decrypted Image');
end    

function enc_dec(img)
    % Encrypt the image
    encrypted_image = chaotic_image_encryption(img);
    
    % Decrypt the image
    decrypted_image = chaotic_image_decryption(encrypted_image);
    
    % Display Original-Encryted-Decryted images
    figure;
    subplot(1, 3, 1);
    imshow(img);
    title('Original Image');
    
    subplot(1, 3, 2);
    imshow(uint8(encrypted_image));
    title('Encrypted Image');
    
    subplot(1, 3, 3);
    imshow(uint8(decrypted_image));
    title('Decrypted Image');
end

% Encrypt Image
function encrypted_image = chaotic_image_encryption(img)
    
    % Step 1: Contourlet Transform
    pfilt = 'pkva';     
    dfilt = 'pkva';     
    nlevs = [0, 2, 3, 4] ;  
    coeff = pdfbdec(double(img), pfilt, dfilt, nlevs);  

    low_freq_subband = coeff{1};  

    % Step 2 : scrabmling
    scrambled_low_freq = arnold_map_scramble(low_freq_subband);
    coeff{1} = scrambled_low_freq;  

    % Step 3: Reconstruct
    scrambled_img = uint8(pdfbrec(coeff, pfilt, dfilt));    

    % Step 4: Encrypt image 
    encrypted_image = uint8(hyperchaotic_diffusion(scrambled_img));
end

% Diffusion Phase
function diffused_img = hyperchaotic_diffusion(img)
    [rows, cols] = size(img);
    num_pixels = rows * cols;

    % Generate chaotic sequences
    [x1_seq, x2_seq, x3_seq, x4_seq] = solve_hyperchaotic_system(num_pixels / 4);

    % Process chaotic sequences using Formula (3)
    x1_seq = uint8(mod(fix((abs(x1_seq(:)) - floor(abs(x1_seq(:)))) * 10^14), rows));
    x2_seq = uint8(mod(fix((abs(x2_seq(:)) - floor(abs(x2_seq(:)))) * 10^14), rows));
    x3_seq = uint8(mod(fix((abs(x3_seq(:)) - floor(abs(x3_seq(:)))) * 10^14), rows));
    x4_seq = uint8(mod(fix((abs(x4_seq(:)) - floor(abs(x4_seq(:)))) * 10^14), rows));
 
    img_vector = reshape(img, [], 1);

    % Encrypt image using Formula (4)
    C = zeros(size(img_vector));
    C(1) = 251; 

    for i = 1:4:num_pixels
        idx = (i - 1) / 4 + 1; 

        if i + 3 <= num_pixels
            C(i) =     bitxor(img_vector(i),     bitxor(x1_seq(idx) , C(i - 1 + (i == 1))));
            C(i + 1) = bitxor(img_vector(i + 1), bitxor(x2_seq(idx) , C(i)));
            C(i + 2) = bitxor(img_vector(i + 2), bitxor(x3_seq(idx) , C(i + 1)));
            C(i + 3) = bitxor(img_vector(i + 3), bitxor(x4_seq(idx) , C(i + 2)));
        end
    end
    
    % Reshape vector back to image dimensions
    diffused_img = reshape(C, rows, cols);
end

% Decrypt Image
% Every action in encryption is reversed.
function decrypted_image = chaotic_image_decryption(encrypted_image)

    % Step 4: Inverse Hyper-Chaotic Diffusion 
    scrambled_img = inverse_hyperchaotic_diffusion(encrypted_image);

    % Step 3: Contourlet Transform 
    pfilt = 'pkva';     
    dfilt = 'pkva';     
    nlevs = [0, 2, 3, 4] ;  
    coeff = pdfbdec(double(scrambled_img), pfilt, dfilt, nlevs);

    scrambled_low_freq = coeff{1};  

    % Step 2: Un Scrambling 
    unscrambled_low_freq = inverse_arnold_map_scramble(scrambled_low_freq);
    coeff{1} = double(unscrambled_low_freq);  

    % Step 1: Reconstruct 
    decrypted_image = uint8(pdfbrec(coeff, pfilt, dfilt));
end


% Inverse Diffusion Phase
function original_img = inverse_hyperchaotic_diffusion(encrypted_img)
    [rows, cols] = size(encrypted_img);
    num_pixels = rows * cols;

    % Generate chaotic sequences 
    [x1_seq, x2_seq, x3_seq, x4_seq] = solve_hyperchaotic_system(num_pixels / 4);

    % Process chaotic sequences using Formula (3)
    x1_seq = uint8(mod(fix((abs(x1_seq(:)) - floor(abs(x1_seq(:)))) * 10^14), rows));
    x2_seq = uint8(mod(fix((abs(x2_seq(:)) - floor(abs(x2_seq(:)))) * 10^14), rows));
    x3_seq = uint8(mod(fix((abs(x3_seq(:)) - floor(abs(x3_seq(:)))) * 10^14), rows));
    x4_seq = uint8(mod(fix((abs(x4_seq(:)) - floor(abs(x4_seq(:)))) * 10^14), rows));

    enc_vector = reshape(encrypted_img, [], 1);

    C = zeros(size(enc_vector));
    C(1) = 251;  

    % Perform inverse diffusion in the reverse order
    for i = num_pixels:-4:4
        idx = floor((i - 1) / 4 + 1);  
                
        % Reverse XOR for each pixel group of four
        if i - 3 > 0            
            enc_vector(i) =     bitxor(enc_vector(i),     bitxor(x4_seq(idx), enc_vector(i - 1)));
            enc_vector(i - 1) = bitxor(enc_vector(i - 1), bitxor(x3_seq(idx), enc_vector(i - 2)));
            enc_vector(i - 2) = bitxor(enc_vector(i - 2), bitxor(x2_seq(idx), enc_vector(i - 3)));
            if i-3 == 1 
                enc_vector(i - 3) = bitxor(enc_vector(i - 3), bitxor(x1_seq(idx), C(1)));
            else 
                enc_vector(i - 3) = bitxor(enc_vector(i - 3), bitxor(x1_seq(idx), enc_vector(i - 4)));
            end
        end
    end

    % Reshape vector back to image dimensions
    original_img = reshape(enc_vector, rows, cols);
end

function [x1_seq, x2_seq, x3_seq, x4_seq] = solve_hyperchaotic_system(num_points)
    
    % Chaotic fonksiyon ilk değer ve sabitler
    x0 = [0.3, -0.2, 1, 1.4];
    a =  36;  
    b =   3;  
    c =  28; 
    d = -16;
    k = 0.2;  

    % Time span for the solver
    t_span = [0, 100];  

    % Solve the differential equations using ode45
    [~, X] = ode45(@(t, X) hyperchaotic_system(t, X, a, b, c, d, k), linspace(t_span(1), t_span(2), num_points), x0);
    
    x1_seq = X(:, 1)';
    x2_seq = X(:, 2)';
    x3_seq = X(:, 3)';
    x4_seq = X(:, 4)';
end

function dXdt = hyperchaotic_system(t, X, a, b, c, d, k)
    % State variables
    x1 = X(1);
    x2 = X(2);
    x3 = X(3);
    x4 = X(4);

    % Differential equations
    dx1 = a * (x2 - x1);
    dx2 = -x1 * x3 + d * x1 + c * x2 - x4;
    dx3 = x1 * x2 - b * x3;
    dx4 = x1 + k;

    dXdt = [dx1; dx2; dx3; dx4];
end

% Arnold Map Scrambling Function
function scrambled_matrix = arnold_map_scramble(matrix)
    [N, ~] = size(matrix);
    scrambled_matrix = matrix;
    iterations = 10;
    for k = 1:iterations
        temp_matrix = scrambled_matrix;
        for x = 1:N
            for y = 1:N
                new_x = mod((x + y-1), N) + 1;
                new_y = mod((x + 2*y-1), N) + 1;
                temp_matrix(new_x, new_y) = scrambled_matrix(x,y);
            end
        end
        scrambled_matrix = temp_matrix;
    end
end

% Inverse Arnold Map Scrambling Function
function unscrambled_matrix = inverse_arnold_map_scramble(matrix)
    [N, ~] = size(matrix);
    unscrambled_matrix = matrix;
    iterations = 10;
    for k = 1:iterations
        temp_matrix = unscrambled_matrix;
        for x = 1:N
            for y = 1:N
                % Inverse Arnold transformation
                new_x = mod((2 * x - y-1), N) + 1;
                new_y = mod((-x + y-1), N) + 1;
                temp_matrix(new_x, new_y) = unscrambled_matrix(x,y);
            end
        end
        unscrambled_matrix = temp_matrix;
    end
end