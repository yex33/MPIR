% Set the path to the directory containing the matrix files
matrix_dir = '/home/joe/Documents/McMaster/CAS_MEng_Ned/MPIR/matrices/testing';

% Set the path to the file containing the list of matrix names
matrix_list_file = '/home/joe/Documents/McMaster/CAS_MEng_Ned/MPIR/report/data/matrix_list.txt';

% Open the matrix list file
fid = fopen(matrix_list_file, 'r');
if fid == -1
    error('Could not open the matrix list file: %s', matrix_list_file);
end

% Read the list of matrix names
matrix_names = textscan(fid, '%s');
matrix_names = matrix_names{1};
fclose(fid);

% Pre-allocate a cell array to store the results
results = cell(length(matrix_names), 7);

% Loop through each matrix
for i = 1:length(matrix_names)
    matrix_name = matrix_names{i};
    matrix_file = fullfile(matrix_dir, [matrix_name, '.mtx']);

    % Read the matrix from the .mtx file
    try
        A = mmread(matrix_file);
    catch ME
        warning('Could not read matrix file %s: %s', matrix_file, ME.message);
        continue;
    end

    % 1. Matrix name
    results{i, 1} = matrix_name;

    % 2. Condition number estimate
    results{i, 2} = condest(A);

    % 3. Matrix size
    [rows, cols] = size(A);
    results{i, 3} = rows; % Assumes square matrix

    % 4. Number of non-zero elements
    num_nonzeros = nnz(A);
    results{i, 4} = num_nonzeros;

    % 5. Non-zero density percentage
    density = (num_nonzeros / (rows * cols)) * 100;
    results{i, 5} = density;

    % 6. Minimum non-zero element
    non_zero_elements = A(A ~= 0);
    results{i, 6} = full(min(abs(non_zero_elements)));

    % 7. Maximum non-zero element
    results{i, 7} = full(max(abs(non_zero_elements)));
end

% Sort the results by condition number (column 2)
sorted_results = sortrows(results, 2);
%%

% Print each row of the sorted results
for i = 1:size(sorted_results, 1)
    % Format each cell using siunitx commands
    matrix_name = sorted_results{i, 1};
    cond_num = sorted_results{i, 2};
    size_val = sorted_results{i, 3};
    nnz_val = sorted_results{i, 4};
    density_val = sorted_results{i, 5};
    min_nnz = sorted_results{i, 6};
    max_nnz = sorted_results{i, 7};

    % fprintf('%s & \\num{%g} & \\num{%d} & \\num{%d} & \\SI{%g}{\\percent} & \\num{%g} & \\num{%g} \\\\\n', ...
    %     matrix_name, cond_num, size_val, nnz_val, density_val, min_nnz, max_nnz);

    % fprintf('\\texttt{%s} & \\num{%.1e} & \\num{%d} & \\num{%d} & \\SI{%.1g}{\\percent} & \\num{%.1e} & \\num{%.1e} \\\\\n', ...
    %     replace(matrix_name, '_', '\_'), cond_num, size_val, nnz_val, density_val, min_nnz, max_nnz);

    fprintf('\\texttt{%s} & %.1e & %.1e & %.1e & %.1g & %.1e & %.1e \\\\\n', ...
        replace(matrix_name, '_', '\_'), cond_num, size_val, nnz_val, density_val, min_nnz, max_nnz);
end
